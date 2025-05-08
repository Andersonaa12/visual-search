import asyncio
import os
from httpx import AsyncClient
from main import app

EXAMPLE_IMG = "https://picsum.photos/seed/visualsearch/256/256"

async def test_search_by_image(monkeypatch):
    # Skip if running outside container (no DB)
    if not os.environ.get("DB_URL"):
        return

    async with AsyncClient(app=app, base_url="http://test") as ac:
        resp = await ac.get(EXAMPLE_IMG)
        img_bytes = resp.content

        res = await ac.post("/search-by-image", files={"file": ("img.jpg", img_bytes, "image/jpeg")})
        assert res.status_code == 200