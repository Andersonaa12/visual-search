import os, io, json, asyncio, logging
from typing import Any, Dict, List

import httpx, numpy as np
from PIL import Image
from tqdm.asyncio import tqdm
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import text

from database import (
    Session, migrate, Product, Category, product_categories
)

GET_IMAGES_URL = os.getenv("GET_IMAGES_URL")
print(f"GET_IMAGES_URL = {GET_IMAGES_URL}")
DIM            = int(os.getenv("VECTOR_DIM", 576))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest")

model = MobileNetV3Small(weights="imagenet", include_top=False, pooling="avg")
assert model.output_shape[-1] == DIM
log.info("📦 MobileNetV3Small cargado (dim=%s)", DIM)
# ---------- helpers ----------
async def fetch_json(c: httpx.AsyncClient, url: str) -> List[Dict[str, Any]]:
    r = await c.get(url, timeout=30)
    r.raise_for_status()
    return r.json()["data"]

async def embed(img_bytes: bytes) -> np.ndarray | None:
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    except Exception:
        return None
    x = preprocess_input(np.expand_dims(np.array(img, dtype="float32"), 0))
    return model.predict(x, verbose=0).flatten().astype("float32")

def vec_to_pg(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:.6f}" for x in v) + "]"

async def ingest() -> None:
    await migrate()

    async with httpx.AsyncClient() as client:
        images = await fetch_json(client, GET_IMAGES_URL)
        log.info("🔎 %s imágenes a procesar", len(images))
        
        async with Session.begin() as sess:
            for item in images:
                pid = item["product_id"]
                cats = item.get("category_ids", [])
                if cats and isinstance(cats[0], dict):
                    cat_ids = [c["id"] for c in cats]
                    for c in cats:
                        await sess.execute(
                            pg_insert(Category)
                            .values(id=c["id"], name=c["name"])
                            .on_conflict_do_nothing(index_elements=["id"])
                        )
                    await sess.execute(
                        pg_insert(Product)
                        .values(id=pid)
                        .on_conflict_do_nothing(index_elements=["id"])
                    )
                    for cid in cat_ids:
                        await sess.execute(
                            pg_insert(product_categories)
                            .values(product_id=pid, category_id=cid)
                            .on_conflict_do_nothing(
                                index_elements=["product_id", "category_id"]
                            )
                        )

        # ---- embeddings en paralelo ----
        sem    = asyncio.Semaphore(20)
        embeds = []
        pbar   = tqdm(total=len(images), desc="Procesando imágenes")

        async def process(item: Dict[str, Any]) -> None:
            async with sem:
                pid, piid = item["product_id"], item["product_image_id"]
                try:
                    r = await client.get(item["image_url"], timeout=60)
                    r.raise_for_status()
                except Exception as exc:
                    log.warning("X (%s/%s) descarga fallida: %s", pid, piid, exc)
                    pbar.update(); return
                vec = await embed(r.content)
                if vec is None:
                    log.warning("!!!  (%s/%s) imagen inválida", pid, piid)
                    pbar.update(); return
                cats = item.get("category_ids", [])
                if cats and isinstance(cats[0], dict):
                    cats = [c["id"] for c in cats]
                embeds.append({
                    "product_id": pid,
                    "product_image_id": piid,
                    "feature_vector": vec_to_pg(vec),
                    "category_ids": cats,
                })
                pbar.update()

        await asyncio.gather(*(process(i) for i in images))
        pbar.close()

    async with Session() as sess:
        await sess.execute(
            text("""
            WITH vals AS (
              SELECT * FROM jsonb_to_recordset(:json) AS
              (product_id int, product_image_id bigint,
               feature_vector vector, category_ids int[])
            )
            INSERT INTO product_image_features
                  (product_id, product_image_id, feature_vector, category_ids)
            SELECT * FROM vals
            ON CONFLICT (product_id, product_image_id)
            DO UPDATE SET
              feature_vector = EXCLUDED.feature_vector,
              category_ids  = EXCLUDED.category_ids
            """),
            {"json": json.dumps(embeds)},
        )
        await sess.commit()
    log.info("🎉 %s embeddings insertados/actualizados", len(embeds))

if __name__ == "__main__":
    asyncio.run(ingest())
