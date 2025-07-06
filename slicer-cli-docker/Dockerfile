FROM python:3.11-alpine as ffmpeg-builder
RUN wget -O ffmpeg.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
RUN mkdir /tmp/ffextracted
RUN tar -xvf ffmpeg.tar.xz -C /tmp/ffextracted --strip-components 1

FROM python:3.11-slim
LABEL maintainer="Kitware, Inc. <kitware@kitware.com>"

# Install system dependencies
RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
    curl \
    ca-certificates \
    git \
    wget \
    build-essential \
    memcached \
    tesseract-ocr \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir \
        "bokeh>=0.12.14,<3" \
        girder-client \
        opencv-python-headless \
        numpy \
        click \
        pytesseract \
        easyocr \
        tqdm && \
    rm -rf /root/.cache/pip/*

# Additional installs
RUN pip install girder-slicer-cli-web

# Copy code and ffmpeg binaries
COPY . $PWD
COPY --from=ffmpeg-builder /tmp/ffextracted/ffmpeg /tmp/ffextracted/ffprobe /usr/local/bin/

ENTRYPOINT ["python", "cli_list.py"]
