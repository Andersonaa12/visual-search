import os
import io
import json
import asyncio
import logging
import warnings
from typing import Any, Dict, List, Tuple
import httpx
import numpy as np
from PIL import Image
from tqdm.asyncio import tqdm
import torch
import open_clip
from transformers import BlipProcessor, BlipForConditionalGeneration
from sklearn.cluster import KMeans
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.sql import text

# Suprimir advertencia de timm
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated")

from app.database import Session, migrate, Product, ProductImageFeature

# Constantes
GET_IMAGES_URL = os.getenv("GET_IMAGES_URL")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", 512))
DL_TIMEOUT = int(os.getenv("DOWNLOAD_TIMEOUT", 60))
BATCH_SIZE = 64
NUM_CATEGORIES = 10

log = logging.getLogger("ingest")
log.setLevel(logging.INFO)

# Modelos
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai", device=device
)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

# Utilidades
def vec_to_pg(v: np.ndarray) -> str:
    """Formato de vector para pgvector."""
    return "[" + ", ".join(f"{x:.6f}" for x in v) + "]"

async def fetch_json(c: httpx.AsyncClient, url: str) -> List[Dict[str, Any]]:
    r = await c.get(url, timeout=DL_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if "data" not in data:
        raise ValueError("JSON debe contener 'data'")
    return data["data"]

def preprocess(img_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return clip_preprocess(img).unsqueeze(0)

def batch_encode_images(img_tensors: List[torch.Tensor]) -> np.ndarray:
    with torch.no_grad():
        emb = clip_model.encode_image(torch.cat(img_tensors).to(device))
    return emb.cpu().numpy().astype("float32")

def batch_encode_texts(texts: List[str]) -> np.ndarray:
    toks = open_clip.tokenize(texts).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(toks)
    return emb.cpu().numpy().astype("float32")

async def generate_caption(img_bytes: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        inputs = blip_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = blip_model.generate(**inputs, max_new_tokens=50)
        return blip_processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        log.debug("Caption error: %s", e)
        return ""

def infer_categories(captions: List[str], embeddings: np.ndarray) -> List[List[str]]:
    """Agrupar Caption para inferir categorías."""
    if not captions:
        return [[] for _ in range(len(captions))]
    
    kmeans = KMeans(n_clusters=min(NUM_CATEGORIES, len(captions)), random_state=42)
    labels = kmeans.fit_predict(embeddings)
    
    # Generar nombres de categoría a partir de los centros de clúster
    category_names = []
    for i in range(min(NUM_CATEGORIES, len(set(labels)))):
        cluster_captions = [captions[j] for j in range(len(captions)) if labels[j] == i]
        # Usa la primera leyenda como un nombre de categoría representativa (simplificado)
        category_names.append(cluster_captions[0].split()[:2])
    
    # Asigna categorías a cada imagen
    categories = []
    for label in labels:
        categories.append(category_names[label] if label < len(category_names) else [])
    return categories

# Ingesta Principal
async def ingest() -> None:
    await migrate()

    async with httpx.AsyncClient() as client:
        images = await fetch_json(client, GET_IMAGES_URL)
        images = images[:70]  # Testeo
        log.info("🔎 %s images received", len(images))

    # Insert products
    async with Session.begin() as sess:
        product_ids = {item["product_id"] for item in images}
        await sess.execute(
            pg_insert(Product)
            .values([{"id": pid} for pid in product_ids])
            .on_conflict_do_nothing(index_elements=["id"])
        )

    # Procesamiento de imágenes por batches
    semaphore = asyncio.Semaphore(20)
    embeddings: List[Dict[str, Any]] = []
    captions: List[str] = []
    img_tensors: List[torch.Tensor] = []
    batch_items: List[Dict[str, Any]] = []

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
                caption = await generate_caption(img_bytes)
                img_tensors.append(tensor)
                captions.append(caption)
                batch_items.append({"pid": pid, "piid": piid, "caption": caption})
            except Exception as exc:
                log.warning("Skipping %s/%s: %s", pid, piid, exc)

    pbar = tqdm(total=len(images), desc="!! Ingesting")
    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i : i + BATCH_SIZE]
        await asyncio.gather(*(process_item(item) for item in batch))
        
        if img_tensors:
            # Codificar imágenes
            img_vecs = batch_encode_images(img_tensors)
            # Codificar captions
            desc_vecs = batch_encode_texts([c if c else "unknown" for c in captions])
            # Codificar categorías
            categories = infer_categories(captions, desc_vecs)
            
            for j, item in enumerate(batch_items):
                embeddings.append(  # Corregido: usar 'embeddings' en lugar de 'embeds'
                    {
                        "product_id": item["pid"],
                        "product_image_id": item["piid"],
                        "feature_vector": vec_to_pg(img_vecs[j]),
                        "description_vector": vec_to_pg(desc_vecs[j]),
                        "categories": categories[j],
                        "description": item["caption"],
                    }
                )
        
        img_tensors.clear()
        captions.clear()
        batch_items.clear()
        pbar.update(len(batch))

    pbar.close()

    # Insert embeddings
    async with Session.begin() as sess:
        await sess.execute(
            text("""
            WITH vals AS (
                SELECT * FROM jsonb_to_recordset(:json) AS
                (product_id int,
                 product_image_id bigint,
                 feature_vector vector,
                 description_vector vector,
                 categories text[],
                 description text)
            )
            INSERT INTO product_image_features
            (product_id, product_image_id,
             feature_vector, description_vector,
             categories, description)
            SELECT product_id, product_image_id,
                   feature_vector, description_vector,
                   categories, description
            FROM vals
            ON CONFLICT (product_id, product_image_id)
            DO UPDATE SET
                feature_vector = EXCLUDED.feature_vector,
                description_vector = EXCLUDED.description_vector,
                categories = EXCLUDED.categories,
                description = EXCLUDED.description
            """),
            {"json": json.dumps(embeddings)},
        )

    log.info("EXITO!!! Ingestion complete: %s records", len(embeddings))

if __name__ == "__main__":
    asyncio.run(ingest())