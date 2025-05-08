# Visual‑Search MVP

End‑to‑end demo that ingests product images from an existing Laravel API, indexes them with `pgvector`,
and exposes a minimal FastAPI service for *search by image*.

## Run locally with Docker

```bash
cp .env.example .env
docker-compose up --build
```

The API will be available at **http://localhost:8000**.

## Endpoints

| Method | Path                | Description                      |
| ------ | ------------------- | -------------------------------- |
| POST   | `/update-image`     | Update or insert a single image  |
| POST   | `/search-by-image`  | Retrieve similar products        |

## Ingestion worker

```bash
docker-compose exec api python -m controller.ingest
```

Progress as well as every processed *product_id* / *product_image_id* pair
will be echoed to the console so you can see what is happening in real‑time.