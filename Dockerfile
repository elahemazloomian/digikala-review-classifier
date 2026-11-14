# API image for the Digikala review classifier (CPU only).
# Model files are NOT inside the image: mount the artifacts folder at /artifacts when you run it.
FROM python:3.11-slim

# PYTHONUNBUFFERED: print logs immediately. PIP_NO_CACHE_DIR: keep the image smaller.
# ARTIFACTS_DIR: where app/main.py looks for the models. HF_HUB_OFFLINE: never try to download from the internet.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ARTIFACTS_DIR=/artifacts \
    HF_HUB_OFFLINE=1

# Package servers. The defaults are the Aliyun mirrors (the normal servers were unreachable on the author's network).
# On another network override them, e.g.:  --build-arg INDEX_URL=https://pypi.org/simple/ --build-arg TORCH_LINKS=https://download.pytorch.org/whl/cpu
ARG INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ARG TORCH_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cpu/

WORKDIR /code

# Install the packages first: this layer is reused while requirements-api.txt does not change.
COPY requirements-api.txt .
RUN pip install --index-url ${INDEX_URL} --find-links ${TORCH_LINKS} -r requirements-api.txt

# Then the code (changes often).
COPY app ./app
COPY src ./src

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
