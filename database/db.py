"""
ORM + migraciones idempotentes.
Se usa SQLAlchemy async puro y ALTER TABLE dinámico
para añadir columnas que falten.
"""
import os
from sqlalchemy import (
    Column, Integer, BigInteger, String,
    ForeignKey, Table, text, insert
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY

DB_URL     = os.getenv("DB_URL")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", 576))

engine  = create_async_engine(DB_URL, pool_size=10, max_overflow=20)
Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base    = declarative_base()

# puente N-a-N
product_categories = Table(
    "product_categories", Base.metadata,
    Column("product_id",  Integer, ForeignKey("products.id",  ondelete="CASCADE")),
    Column("category_id", Integer, ForeignKey("categories.id", ondelete="CASCADE")),
    schema=None,
)

class Category(Base):
    __tablename__ = "categories"

    id          = Column(Integer, primary_key=True)
    name        = Column(String(80), nullable=False)
    description = Column(String)
    parent_id   = Column(Integer, ForeignKey("categories.id"))
    # sin lazy-load ⇒ evitamos IO inesperado
    children    = relationship("Category",
                               backref="parent",
                               remote_side=[id],
                               lazy="selectin")

class Product(Base):
    __tablename__ = "products"

    id   = Column(Integer, primary_key=True)
    name = Column(String)
    categories = relationship("Category",
                              secondary=product_categories,
                              lazy="selectin")

class ProductImageFeature(Base):
    __tablename__ = "product_image_features"

    id               = Column(BigInteger, primary_key=True)
    product_id       = Column(Integer, ForeignKey("products.id", ondelete="CASCADE"))
    product_image_id = Column(BigInteger)
    feature_vector   = Column(Vector(VECTOR_DIM))
    category_ids     = Column(ARRAY(Integer))

# ---------------- migración idempotente ----------------
async def migrate() -> None:
    async with engine.begin() as conn:
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)

        await conn.execute(text("""
            ALTER TABLE categories
            ADD COLUMN IF NOT EXISTS description TEXT,
            ADD COLUMN IF NOT EXISTS parent_id INT,
            ADD COLUMN IF NOT EXISTS created_at  TIMESTAMPTZ DEFAULT now(),
            ADD COLUMN IF NOT EXISTS updated_at  TIMESTAMPTZ DEFAULT now();
        """))

        await conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE  conname = 'product_categories_pk'
                ) THEN
                    ALTER TABLE product_categories
                    ADD CONSTRAINT product_categories_pk
                    PRIMARY KEY (product_id, category_id);
                END IF;
            END$$;
        """))

