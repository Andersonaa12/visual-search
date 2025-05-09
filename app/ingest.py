import os, io, json, asyncio, logging
from typing import Any, Dict, List

import httpx, numpy as np
from PIL import Image
from tqdm.asyncio import tqdm

import torch
import open_clip
from transformers import BlipProcessor, BlipForConditionalGeneration
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import text

from app.database import Session, migrate, Product, Category, product_categories, ProductImageFeature

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
GET_IMAGES_URL = os.getenv("GET_IMAGES_URL")
DIM = int(os.getenv("VECTOR_DIM", 512))
DL_TIMEOUT = int(os.getenv("DOWNLOAD_TIMEOUT", 60))

log = logging.getLogger("ingest")
log.setLevel(logging.INFO)

# ---------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai", device=device
)

blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
blip_model = (
    BlipForConditionalGeneration
    .from_pretrained("Salesforce/blip-image-captioning-base")
    .to(device)
)

# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------
def vec_to_pg(v: np.ndarray) -> str:
    """Formatea el vector para pgvector."""
    return "[" + ", ".join(f"{x:.6f}" for x in v) + "]"

async def fetch_json(c: httpx.AsyncClient, url: str) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
    r = await c.get(url, timeout=DL_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if "data" not in data or "categories" not in data:
        raise ValueError("JSON debe contener 'data' y 'categories'")
    return data["data"], data["categories"]

def preprocess(img_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return clip_preprocess(img).unsqueeze(0)

def batch_encode(img_tensors: List[torch.Tensor]) -> np.ndarray:
    with torch.no_grad():
        emb = clip_model.encode_image(torch.cat(img_tensors).to(device))
    return emb.cpu().numpy().astype("float32")

async def generate_caption(img_bytes: bytes) -> str | None:
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        inputs = blip_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = blip_model.generate(**inputs, max_new_tokens=30)
        return blip_processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        log.debug("caption fail: %s", e)
        return None

# ---------------------------------------------------------------------
# Ingest principal
# ---------------------------------------------------------------------
async def ingest() -> None:
    await migrate()

    async with httpx.AsyncClient() as client:
        images, categories = await fetch_json(client, GET_IMAGES_URL)
        images = images[:1000]  # Descomentar para pruebas
    log.info("🔎 %s imágenes recibidas", len(images))

    # 1) Alta de categorías / productos
    async with Session.begin() as sess:
        # Insertar categorías desde el diccionario de categorías
        category_list = [{"id": int(k), "name": v} for k, v in categories.items()]
        await sess.execute(
            pg_insert(Category)
            .values(category_list)
            .on_conflict_do_nothing(index_elements=["id"])
        )

        # Insertar productos
        product_ids = {item["product_id"] for item in images}
        await sess.execute(
            pg_insert(Product)
            .values([{"id": pid} for pid in product_ids])
            .on_conflict_do_nothing(index_elements=["id"])
        )

        # Insertar relaciones producto-categoría
        product_category_pairs = set()
        for item in images:
            pid = item["product_id"]
            cat_ids = item.get("category_ids", [])
            for cid in cat_ids:
                product_category_pairs.add((pid, cid))
        await sess.execute(
            pg_insert(product_categories)
            .values([{"product_id": pid, "category_id": cid} for pid, cid in product_category_pairs])
            .on_conflict_do_nothing()
        )

    # 2) Descarga concurrente + embeddings
    batch_size = 32
    semaphore = asyncio.Semaphore(20)
    embeds: List[Dict[str, Any]] = []
    pbar = tqdm(total=len(images), desc="!!!  Ingestando")

    async def process_item(item: Dict[str, Any]) -> None:
        async with semaphore:
            pid = item["product_id"]
            piid = item.get("product_image_id")
            try:
                async with httpx.AsyncClient() as c:
                    r = await c.get(item["image_url"], timeout=DL_TIMEOUT)
                    r.raise_for_status()
                img_bytes = r.content
                tensor = preprocess(img_bytes)
                vec = batch_encode([tensor])[0]
                caption = await generate_caption(img_bytes)
                cat_ids = item.get("category_ids", [])
                embeds.append(
                    dict(
                        product_id=pid,
                        product_image_id=piid,
                        feature_vector=vec_to_pg(vec),
                        category_ids=cat_ids,
                        description=caption,
                    )
                )
            except Exception as exc:
                log.warning("skip %s/%s: %s", pid, piid, exc)
            finally:
                pbar.update()

    await asyncio.gather(*(process_item(it) for it in images))
    pbar.close()

    async with Session.begin() as sess:
        await sess.execute(
            text("""
            WITH vals AS (
              SELECT * FROM jsonb_to_recordset(:json) AS
              (product_id int,
               product_image_id bigint,
               feature_vector vector,
               category_ids int[],
               description text)
            )
            INSERT INTO product_image_features
                  (product_id, product_image_id,
                   feature_vector, category_ids, description)
            SELECT product_id, product_image_id,
                   feature_vector, category_ids, description
            FROM vals
            ON CONFLICT (product_id, product_image_id)
            DO UPDATE SET
              feature_vector = EXCLUDED.feature_vector,
              category_ids   = EXCLUDED.category_ids,
              description    = EXCLUDED.description
            """),
            {"json": json.dumps(embeds)},
        )

    log.info("🎉 Ingesta completa: %s registros", len(embeds))

if __name__ == "__main__":
    asyncio.run(ingest())