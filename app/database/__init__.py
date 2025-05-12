import os
import logging
from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    ForeignKey,
    Text,
    ARRAY,
    UniqueConstraint,
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY

# Configuration
DB_URL = os.getenv("DB_URL")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", 512))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("database")

engine = create_async_engine(DB_URL, pool_size=20, max_overflow=40)
Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# Models
class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True)
    name = Column(String(80), nullable=True)

class ProductImageFeature(Base):
    __tablename__ = "product_image_features"

    id = Column(BigInteger, primary_key=True)
    product_id = Column(Integer, ForeignKey("products.id", ondelete="CASCADE"))
    product_image_id = Column(BigInteger, nullable=True)
    feature_vector = Column(Vector(VECTOR_DIM))
    description_vector = Column(Vector(VECTOR_DIM))
    categories = Column(ARRAY(Text), nullable=True)
    description = Column(Text)

    __table_args__ = (
        UniqueConstraint(
            "product_id",
            "product_image_id",
            name="product_image_features_uq",
        ),
    )

# Migration (idempotent)
async def migrate() -> None:
    from sqlalchemy.sql import text

    async with engine.begin() as conn:
        log.info("!! Migrating database")

        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)

        # HNSW indices for pgvector
        await conn.execute(
            text("""
            CREATE INDEX IF NOT EXISTS product_image_features_hnsw_img
            ON product_image_features
            USING hnsw (feature_vector vector_cosine_ops);
            """)
        )
        await conn.execute(
            text("""
            CREATE INDEX IF NOT EXISTS product_image_features_hnsw_desc
            ON product_image_features
            USING hnsw (description_vector vector_cosine_ops);
            """)
        )

        # Ensure constraints
        await conn.execute(
            text("""
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
            END$$;
            """)
        )

        log.info("!! Migration completed")