import os, io, json, base64, logging
from typing import List, Optional

import numpy as np
import torch, open_clip, umap
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import matplotlib;  matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import BlipProcessor, BlipForConditionalGeneration
from sqlalchemy import text

from app.database import Session, migrate, ProductImageFeature

# —————————————————— config ——————————————————
DIM             = int(os.getenv("VECTOR_DIM", 512))
TOP_K           = int(os.getenv("TOP_K", 10))
DIST_THRESHOLD  = float(os.getenv("DIST_THRESHOLD", "0.28"))
W_IMG_DEFAULT   = float(os.getenv("W_IMG_DEFAULT", "0.7"))
W_TXT_DEFAULT   = float(os.getenv("W_TXT_DEFAULT", "0.3"))

log = logging.getLogger("main");  log.setLevel(logging.INFO)
app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
blip_proc  = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# —————————————————— helpers ——————————————————
def vec_to_pg(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:.6f}" for x in v) + "]"

def embed_clip(img: Image.Image) -> np.ndarray:
    with torch.no_grad():
        return (clip_model.encode_image(clip_preprocess(img).unsqueeze(0).to(device))
                .cpu().numpy().flatten().astype("float32"))

async def gen_caption(img: Image.Image) -> str:
    inp = blip_proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip_model.generate(**inp, max_new_tokens=40)
    return blip_proc.decode(out[0], skip_special_tokens=True)

def embed_text(text: str) -> np.ndarray:
    toks = open_clip.tokenize([text]).to(device)
    with torch.no_grad():
        return (clip_model.encode_text(toks)
                .cpu().numpy().flatten().astype("float32"))

try:
    umap_model = umap.UMAP(n_components=2, metric="cosine", n_neighbors=15, min_dist=0.1)
except Exception:
    from sklearn.decomposition import PCA;  umap_model = PCA(n_components=2)

def scatter_plot(emb: np.ndarray, labels: list[str]) -> str:
    plt.figure(figsize=(10, 8)); plt.scatter(emb[:, 0], emb[:, 1], alpha=0.6)
    for i, lbl in enumerate(labels): plt.annotate(lbl, (emb[i, 0], emb[i, 1]))
    buf = io.BytesIO(); plt.savefig(buf, format="png"); plt.close(); buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# —————————————————— fastapi lifecycle ——————————————————
@app.on_event("startup")
async def on_startup(): await migrate()

# —————————————————— pydantic ——————————————————
class Match(BaseModel):
    product_id: int
    product_image_id: Optional[int]
    description: Optional[str]
    category_ids: Optional[List[int]]
    distance: float

class SearchResponse(BaseModel):
    query_caption: str
    best_match: Match
    similar_products: List[Match]
    graph_url: Optional[str] = None

# —————————————————— update-image ——————————————————
@app.post("/update-image")
async def update_image(
    product_id: int = Form(...),
    product_image_id: Optional[int] = Form(None),
    category_ids: str = Form("[]"),
    file: UploadFile = File(...),
):
    try:
        img = Image.open(file.file).convert("RGB")
        img_vec      = embed_clip(img)
        cap          = await gen_caption(img)
        cap_vec      = embed_text(cap)

        try:
            cats = json.loads(category_ids)
            if not isinstance(cats, list): raise ValueError
        except ValueError:
            return JSONResponse(status_code=400, content={"error": "`category_ids` debe ser lista JSON"})

        async with Session.begin() as sess:
            await sess.execute(text("""
                INSERT INTO product_image_features
                      (product_id, product_image_id,
                       feature_vector, description_vector,
                       category_ids, description)
                VALUES (:pid, :piid, :fv, :dv, :cats, :cap)
                ON CONFLICT (product_id, product_image_id)
                DO UPDATE SET
                  feature_vector     = EXCLUDED.feature_vector,
                  description_vector = EXCLUDED.description_vector,
                  category_ids       = EXCLUDED.category_ids,
                  description        = EXCLUDED.description
            """), dict(pid=product_id, piid=product_image_id,
                       fv=vec_to_pg(img_vec), dv=vec_to_pg(cap_vec),
                       cats=cats, cap=cap))
        return {"status": "ok"}
    except Exception as e:
        log.exception("update_image");  return JSONResponse(status_code=500, content={"error": str(e)})

# —————————————————— search-by-image ——————————————————
@app.post("/search-by-image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(...),
    k: int = Form(TOP_K),
    text_query: str = Form(""),
    w_img: float = Form(W_IMG_DEFAULT),
    w_txt: float = Form(W_TXT_DEFAULT),
):
    try:
        # 1) obtener embeddings consulta
        img = Image.open(file.file).convert("RGB")
        img_vec     = embed_clip(img)
        q_caption   = await gen_caption(img)              # descripción generada
        cap_vec     = embed_text(q_caption)

        # si el usuario añade texto, lo mezclamos con la caption
        if text_query.strip():
            user_vec = embed_text(text_query.strip())
            cap_vec  = (cap_vec + user_vec) / 2

        #vector híbrido
        query_vec = (w_img * img_vec + w_txt * cap_vec) / (w_img + w_txt)

        #SQL con umbral de distancia
        async with Session() as sess:
            rows = (await sess.execute(text("""
                SELECT product_id, product_image_id, description,
                       category_ids, feature_vector,
                       feature_vector <-> :q AS dist
                FROM product_image_features
                WHERE feature_vector <-> :q < :th
                ORDER BY dist
                LIMIT :lim
            """), dict(q=vec_to_pg(query_vec), th=DIST_THRESHOLD, lim=k))).all()

        if not rows:
            return JSONResponse(status_code=404, content={"error": "Sin resultados similares"})

        best = rows[0]
        best_match = Match(product_id=best[0], product_image_id=best[1],
                           description=best[2], category_ids=best[3], distance=float(best[5]))
        similars   = [Match(product_id=r[0], product_image_id=r[1],
                            description=r[2], category_ids=r[3], distance=float(r[5]))
                      for r in rows[1:]]

        #grafico UMAP
        graph_url = None
        try:
            emb = np.vstack([r[4] for r in rows])
            graph_url = scatter_plot(umap_model.fit_transform(emb),
                                     [f"{r[0]}/{r[1]}" for r in rows])
        except Exception: pass

        return SearchResponse(query_caption=q_caption,
                              best_match=best_match,
                              similar_products=similars,
                              graph_url=graph_url)
    except Exception as e:
        log.exception("search_by_image");  return JSONResponse(status_code=500, content={"error": str(e)})
