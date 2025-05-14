import os
import io
import json
import logging
import base64
from typing import List, Optional
import numpy as np
from PIL import Image
import torch
import open_clip
import umap
import matplotlib

# Configurar logging detallado
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from sqlalchemy.sql import text

from app.database import Session, migrate, ProductImageFeature

# Constantes
VECTOR_DIM = int(os.getenv("VECTOR_DIM", 512))
TOP_K = int(os.getenv("TOP_K", 10))
DIST_THRESHOLD = float(os.getenv("DIST_THRESHOLD", 0.28))
W_IMG_DEFAULT = float(os.getenv("W_IMG_DEFAULT", 0.7))
W_TXT_DEFAULT = float(os.getenv("W_TXT_DEFAULT", 0.3))

log.info("Iniciando la aplicación FastAPI")
app = FastAPI()

log.info("Detectando dispositivo para PyTorch")
device = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"Dispositivo seleccionado: {device}")

log.info("Cargando modelo CLIP ViT-B-32")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai", device=device
)
log.info("Modelo CLIP cargado exitosamente")

log.info("Cargando procesador BLIP")
blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
log.info("Procesador BLIP cargado exitosamente")

log.info("Cargando modelo BLIP")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)
log.info("Modelo BLIP cargado exitosamente")

# Helpers
def vec_to_pg(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:.6f}" for x in v) + "]"

def embed_clip(img: Image.Image) -> np.ndarray:
    with torch.no_grad():
        return (
            clip_model.encode_image(
                clip_preprocess(img).unsqueeze(0).to(device)
            )
            .cpu()
            .numpy()
            .flatten()
            .astype("float32")
        )

async def gen_caption(img: Image.Image) -> str:
    inp = blip_proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip_model.generate(**inp, max_new_tokens=50)
    return blip_proc.decode(out[0], skip_special_tokens=True)

def embed_text(text: str) -> np.ndarray:
    toks = open_clip.tokenize([text]).to(device)
    with torch.no_grad():
        return (
            clip_model.encode_text(toks)
            .cpu()
            .numpy()
            .flatten()
            .astype("float32")
        )

log.info("Inicializando modelo UMAP")
try:
    umap_model = umap.UMAP(
        n_components=2, metric="cosine", n_neighbors=15, min_dist=0.1
    )
    log.info("Modelo UMAP inicializado exitosamente")
except Exception as e:
    log.warning(f"Fallo al inicializar UMAP: {str(e)}. Usando PCA como respaldo.")
    from sklearn.decomposition import PCA
    umap_model = PCA(n_components=2)
    log.info("Modelo PCA inicializado como respaldo")

def scatter_plot(emb: np.ndarray, labels: List[str]) -> str:
    plt.figure(figsize=(10, 8))
    plt.scatter(emb[:, 0], emb[:, 1], alpha=0.6)
    for i, lbl in enumerate(labels):
        plt.annotate(lbl, (emb[i, 0], emb[i, 1]))
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# FastAPI lifecycle
@app.on_event("startup")
async def on_startup():
    log.info("Ejecutando migración de la base de datos")
    await migrate()
    log.info("Migración de la base de datos completada")

# Pydantic models
class Match(BaseModel):
    product_id: int
    product_image_id: Optional[int]
    description: Optional[str]
    categories: Optional[List[str]]
    distance: float

class SearchResponse(BaseModel):
    query_caption: str
    best_match: Match
    similar_products: List[Match]
    graph_url: Optional[str] = None

# Actualizar Imagen - Endpoint
@app.post("/update-image")
async def update_image(
    product_id: int = Form(...),
    product_image_id: Optional[int] = Form(None),
    file: UploadFile = File(...),
):
    try:
        img = Image.open(file.file).convert("RGB")
        img_vec = embed_clip(img)
        caption = await gen_caption(img)
        desc_vec = embed_text(caption)

        async with Session.begin() as sess:
            await sess.execute(
                text("""
                INSERT INTO product_image_features
                (product_id, product_image_id,
                 feature_vector, description_vector,
                 categories, description)
                VALUES (:pid, :piid, :fv, :dv, :cats, :cap)
                ON CONFLICT (product_id, product_image_id)
                DO UPDATE SET
                    feature_vector = EXCLUDED.feature_vector,
                    description_vector = EXCLUDED.description_vector,
                    categories = EXCLUDED.categories,
                    description = EXCLUDED.description
                """),
                {
                    "pid": product_id,
                    "piid": product_image_id,
                    "fv": vec_to_pg(img_vec),
                    "dv": vec_to_pg(desc_vec),
                    "cats": caption.split()[:2],  # Categoria simplificada
                    "cap": caption,
                },
            )
        return {"status": "ok"}
    except Exception as e:
        log.exception("update_image")
        return JSONResponse(status_code=500, content={"error": str(e)})

# Búsqueda x Imagen
@app.post("/search-by-image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(...),
    k: int = Form(TOP_K),
    text_query: str = Form(""),
    w_img: float = Form(W_IMG_DEFAULT),
    w_txt: float = Form(W_TXT_DEFAULT),
):
    try:
        # Obtener query embeddings
        img = Image.open(file.file).convert("RGB")
        img_vec = embed_clip(img)
        q_caption = await gen_caption(img)
        cap_vec = embed_text(q_caption)

        # Mezcla texto de consulta de usuario si se proporciona
        if text_query.strip():
            user_vec = embed_text(text_query.strip())
            cap_vec = (cap_vec + user_vec) / 2

        # Vector de consulta híbrido
        query_vec = (w_img * img_vec + w_txt * cap_vec) / (w_img + w_txt)

        # Query database
        async with Session() as sess:
            rows = (
                await sess.execute(
                    text("""
                    SELECT  product_id,
                            product_image_id,
                            description,
                            categories,
                            feature_vector,
                            ((feature_vector    <-> CAST(:q_img  AS vector))
                            + 
                            (description_vector <-> CAST(:q_desc AS vector))
                            ) / 2 AS dist
                    FROM    product_image_features
                    WHERE   ((feature_vector    <-> CAST(:q_img  AS vector))
                            + 
                            (description_vector <-> CAST(:q_desc AS vector))
                            ) / 2 < :th
                    ORDER BY dist
                    LIMIT  :lim
                    """),
                    {
                        "q_img": vec_to_pg(img_vec),
                        "q_desc": vec_to_pg(cap_vec),
                        "th": 99.28,
                        "lim": k,
                    },
                )
            ).all()



        if not rows:
            return JSONResponse(
                status_code=404, content={"error": "No similar results"}
            )

        # Formateo de results
        best = rows[0]
        best_match = Match(
            product_id=best[0],
            product_image_id=best[1],
            description=best[2],
            categories=best[3],
            distance=float(best[5]),
        )
        similar = [
            Match(
                product_id=r[0],
                product_image_id=r[1],
                description=r[2],
                categories=r[3],
                distance=float(r[5]),
            )
            for r in rows[1:]
        ]

        # UMAP/PCA visualización
        graph_url = None
        try:
            emb = np.vstack([r[4] for r in rows])
            graph_url = scatter_plot(
                umap_model.fit_transform(emb), [f"{r[0]}/{r[1]}" for r in rows]
            )
        except Exception:
            pass

        return SearchResponse(
            query_caption=q_caption,
            best_match=best_match,
            similar_products=similar,
            graph_url=graph_url,
        )
    except Exception as e:
        log.exception("search_by_image")
        return JSONResponse(status_code=500, content={"error": str(e)})