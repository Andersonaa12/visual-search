FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app \
    KERAS_HOME=/root/.keras

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Descargamos los pesos de MobileNetV3 una sola vez
RUN python - <<'PY'
from keras.utils import get_file
get_file(
    "weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5",
    origin="https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v3/weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5",
    cache_subdir="models",
)
PY

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
