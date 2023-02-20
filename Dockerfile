
# CUDA 11.7 (Runtime Ubuntu 20.04) をベースイメージとして利用
FROM nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu20.04

# タイムゾーンを東京に設定
ENV TZ=Asia/Tokyo

# apt-get に対話的に設定を確認されないための設定
ENV DEBIAN_FRONTEND=noninteractive

# Python 3.10・FFmpeg・SoX・ESPnet の動作に必要な各種ソフトのインストール
## ref: https://github.com/espnet/espnet/blob/master/docker/prebuilt/runtime.dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git software-properties-common tzdata wget && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        curl \
        python3.10 \
        python3.10-distutils \
        python3.10-venv \
        ffmpeg \
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

# コンテナ内での作業ディレクトリを指定
WORKDIR /code/

# pip をインストール
## python3-pip だと Python 3.8 ベースの pip がインストールされるため、get-pip.py でインストールする
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.10

# pipenv をインストール
RUN pip install pipenv

# 依存パッケージリスト (Pipfile/Pipfile.lock) だけをコピー
COPY ./Pipfile ./Pipfile.lock /code/

# 依存パッケージを pipenv でインストール
## 仮想環境 (.venv) をプロジェクト直下に作成する
ENV PIPENV_VENV_IN_PROJECT true
RUN pipenv sync

# /root/.keras/ から /root/.cache/ にシンボリックリンクを貼る
## ホスト側の .cache/ に inaSpeechSegmenter の学習済みモデルを保存できるようにする
RUN cd /root/ && ln -s .cache/ .keras

# ソースコードをコピー
COPY ./ /code/

# Aivis.py をエントリーポイントとして指定
ENTRYPOINT ["pipenv", "run", "python", "Aivis.py"]
