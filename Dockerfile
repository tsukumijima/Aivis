
# CUDA 12.1.1 (CUDNN8 Runtime Ubuntu 20.04) をベースイメージとして利用
FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04

# タイムゾーンを東京に設定
ENV TZ=Asia/Tokyo

# apt-get に対話的に設定を確認されないための設定
ENV DEBIAN_FRONTEND=noninteractive

# Python 3.11 と動作に必要な各種ソフトのインストール
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git software-properties-common tzdata wget && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        curl \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        python3.11-venv \
        sox \
        automake \
        autoconf \
        apt-utils \
        bc \
        build-essential \
        cmake \
        flac \
        gawk \
        gfortran \
        libboost-all-dev \
        libtool \
        libbz2-dev \
        liblzma-dev \
        libsndfile1-dev \
        patch \
        python2.7 \
        subversion \
        unzip \
        zip \
        zlib1g-dev && \
    apt-get -y autoremove && \
    apt-get -y clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/*

# FFmpeg 6.0 をインストール
RUN curl -LO \
    https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n6.0-latest-linux64-gpl-shared-6.0.tar.xz && \
    tar -xvf ffmpeg-n6.0-latest-linux64-gpl-shared-6.0.tar.xz && \
    cp -ar ffmpeg-n6.0-latest-linux64-gpl-shared-6.0/bin/* /usr/bin/ && \
    cp -ar ffmpeg-n6.0-latest-linux64-gpl-shared-6.0/lib/* /usr/lib/ && \
    rm -rf ffmpeg-n6.0-latest-linux64-gpl-shared-6.0 && \
    rm -rf ffmpeg-n6.0-latest-linux64-gpl-shared-6.0.tar.xz

# コンテナ内での作業ディレクトリを指定
WORKDIR /code/

# pip をインストール
## python3-pip だと Python 3.8 ベースの pip がインストールされるため、get-pip.py でインストールする
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.11

# poetry をインストール
RUN pip install poetry

# Poetry の依存パッケージリストだけをコピー
COPY ./pyproject.toml ./poetry.lock ./poetry.toml /code/

# 依存パッケージを poetry でインストール
RUN poetry env use 3.11 && \
    poetry install --only main --no-root

# /root/.keras/ から /root/.cache/ にシンボリックリンクを貼る
## ホスト側の .cache/ に inaSpeechSegmenter の学習済みモデルを保存できるようにする
RUN cd /root/ && ln -s .cache/ .keras

# ソースコードをコピー
COPY ./ /code/

# Aivis.py をエントリーポイントとして指定
ENTRYPOINT ["poetry", "run", "python", "Aivis.py"]
