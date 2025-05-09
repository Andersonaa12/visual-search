import os
import logging
from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    ForeignKey,
    Table,
    UniqueConstraint,
    text,
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY

# ---------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------
DB_URL = os.getenv("DB_URL")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", 512))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("database")

engine = create_async_engine(DB_URL, pool_size=10, max_overflow=20)
Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# ---------------------------------------------------------------------
# Modelo relacional
# ---------------------------------------------------------------------
product_categories = Table(
    "product_categories",
    Base.metadata,
    Column(
        "product_id",
        Integer,
        ForeignKey("products.id", ondelete="CASCADE"),
    ),
    Column(
        "category_id",
        Integer,
        ForeignKey("categories.id", ondelete="CASCADE"),
    ),
    UniqueConstraint("product_id", "category_id", name="product_categories_uq"),
)

class Category(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True)
    name = Column(String(80), nullable=False)
    description = Column(String)
    parent_id = Column(Integer, ForeignKey("categories.id"))
    children = relationship(
        "Category",
        backref="parent",
        remote_side=[id],
        lazy="selectin",
    )

class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    categories = relationship(
        "Category",
        secondary=product_categories,
        lazy="selectin",
    )

class ProductImageFeature(Base):
    __tablename__ = "product_image_features"

    id = Column(BigInteger, primary_key=True)
    product_id = Column(Integer, ForeignKey("products.id", ondelete="CASCADE"))
    product_image_id = Column(BigInteger, nullable=True)
    feature_vector = Column(Vector(VECTOR_DIM))
    category_ids = Column(ARRAY(Integer))
    description = Column(String)

    __table_args__ = (
        UniqueConstraint(
            "product_id",
            "product_image_id",
            name="product_image_features_uq",
        ),
    )

# ---------------------------------------------------------------------
# Migración (idempotente)
# ---------------------------------------------------------------------
async def migrate() -> None:
    async with engine.begin() as conn:
        log.info("!!! Migrando base de datos")

        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)

        # indice HNSW para pgvector
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS product_image_features_hnsw
            ON product_image_features
            USING hnsw (feature_vector vector_cosine_ops);
        """))

        # Asegura las constraints
        await conn.execute(text("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'product_image_features_uq'
            ) THEN
                ALTER TABLE product_image_features
                ADD CONSTRAINT product_image_features_uq
                UNIQUE (product_id, product_image_id);
            END IF;

            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'product_categories_uq'
            ) THEN
                ALTER TABLE product_categories
                ADD CONSTRAINT product_categories_uq
                UNIQUE (product_id, category_id);
            END IF;
        END$$;
        """))

    log.info("OK!!! Migración finalizada")