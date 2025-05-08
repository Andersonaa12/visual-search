"""
API FastAPI:
- /update-image  → upsert de un solo vector
- /search-by-image → busqueda semantica con IA + filtro opcional por categoría
"""
import io, json, os, logging
from typing import Any, List, Dict

import numpy as np
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from sqlalchemy import text
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing import image as kimage

from database import Session, migrate

DIM = int(os.getenv("VECTOR_DIM", 576))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("api")

# ---------- modelo visión ---------------------------------------------------
model = MobileNetV3Small(weights="imagenet", include_top=False, pooling="avg")
assert model.output_shape[-1] == DIM, f"Expected dim {DIM}, got {model.output_shape[-1]}"
log.info("📦 MobileNetV3Small cargado (dim=%s)", DIM)

# ---------- FastAPI ---------------------------------------------------------
app = FastAPI(title="Visual-Search MVP")

@app.on_event("startup")
async def _startup() -> None:
    await migrate()
    log.info("OK!!! Base de datos lista")

def embed(img_bytes: bytes) -> np.ndarray:
    img = kimage.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
    x   = kimage.img_to_array(img, dtype="float32")
    x   = preprocess_input(np.expand_dims(x, 0))
    return model.predict(x, verbose=0).flatten().astype("float32")

def vec_to_sql(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:.6f}" for x in v) + "]"

# ---------- rutas -----------------------------------------------------------
@app.post("/update-image")
async def update_image(
    product_id: int        = Form(...),
    product_image_id: int | None = Form(None),
    category_ids: str      = Form("[]"), 
    file: UploadFile       = Form(...),
):
    """Upsert del vector de *una* imagen subido desde Laravel."""
    vec        = embed(await file.read())
    cats: List[int] = json.loads(category_ids)

    async with Session() as s:
        await s.execute(
            text("""
            INSERT INTO product_image_features
                  (product_id, product_image_id, feature_vector, category_ids)
            VALUES (:pid, :piid, :vec, :cats)
            ON CONFLICT (product_id, product_image_id)
            DO UPDATE SET
                feature_vector = EXCLUDED.feature_vector,
                category_ids  = EXCLUDED.category_ids
            """),
            {"pid": product_id, "piid": product_image_id,
             "vec": vec.tolist(), "cats": cats},
        )
        await s.commit()

    log.info("⬆️  upserted product=%s image=%s", product_id, product_image_id)
    return {"status": "ok"}


@app.post("/search-by-image")
async def search_by_image(
    file: UploadFile,
    k: int = Form(10),
    category_id: int | None = Form(None),
):
    """
    Devuelve los *k* productos más similares semánticamente a la imagen subida.
    Si llega `category_id` ⇒ filtro taxonómico adicional.
    """
    qvec_sql = vec_to_sql(embed(await file.read()))

    where  = ""
    params: Dict[str, Any] = {"qvec": qvec_sql, "k": k}
    if category_id:
        where = "WHERE :cid::int = ANY(category_ids)"
        params["cid"] = category_id

    sql = f"""
        SELECT
          product_id,
          product_image_id,
          feature_vector <-> CAST(:qvec AS vector) AS distance
        FROM product_image_features
        {where}
        ORDER BY feature_vector <-> CAST(:qvec AS vector)
        LIMIT :k
    """

    async with Session() as s:
        rows = await s.execute(text(sql), params)
        results = [
            {"product_id": r[0], "product_image_id": r[1], "distance": float(r[2])}
            for r in rows
        ]

    if results:
        bm = results[0]
        log.info("🔍 best_match pid=%s img=%s dist=%.4f", bm["product_id"], bm["product_image_id"], bm["distance"])

    return JSONResponse({"best_match": results[0] if results else None,
                         "similar_products": results})
