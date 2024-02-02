
# CUDA 12.1.1 (CUDNN8 Runtime Ubuntu 20.04) をベースイメージとして利用
FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04

# タイムゾーンを東京に設定
ENV TZ=Asia/Tokyo

# apt-get に対話的に設定を確認されないための設定
ENV DEBIAN_FRONTEND=noninteractive

# Python 3.11 と動作に必要な各種ソフトのインストール
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git software-properties-common tzdata && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        python3.11-venv \
        build-essential \
        cmake && \
    apt-get -y autoremove && \
    apt-get -y clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/*

# FFmpeg 6.0 をインストール
RUN curl -LO \
    https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2023-11-30-12-55/ffmpeg-n6.0.1-linux64-gpl-shared-6.0.tar.xz && \
    tar -xvf ffmpeg-n6.0.1-linux64-gpl-shared-6.0.tar.xz && \
    cp -ar ffmpeg-n6.0.1-linux64-gpl-shared-6.0/bin/* /usr/bin/ && \
    cp -ar ffmpeg-n6.0.1-linux64-gpl-shared-6.0/lib/* /usr/lib/ && \
    rm -rf ffmpeg-n6.0.1-linux64-gpl-shared-6.0 && \
    rm -rf ffmpeg-n6.0.1-linux64-gpl-shared-6.0.tar.xz

# コンテナ内での作業ディレクトリを指定
WORKDIR /code/

# pip をインストール
## python3-pip だと古い Python 向けの pip がインストールされるため、get-pip.py でインストールする
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.11

# Poetry をインストール
RUN pip install poetry

# Poetry の依存パッケージリストだけをコピー
COPY ./pyproject.toml ./poetry.lock ./poetry.toml /code/

# 依存パッケージを Poetry でインストール
RUN poetry env use 3.11 && \
    poetry install --only main --no-root

# 事前に OpenJTalk の辞書をダウンロードしておく
## 通常は動的に行われるが、Docker イメージを再ビルドした際に毎回ダウンロードされるのを防ぐ
RUN cd /code/.venv/lib/python3.11/site-packages/pyopenjtalk/ && \
    curl -LO https://github.com/r9y9/open_jtalk/releases/download/v1.11.1/open_jtalk_dic_utf_8-1.11.tar.gz && \
    tar -xvf open_jtalk_dic_utf_8-1.11.tar.gz && \
    rm -rf open_jtalk_dic_utf_8-1.11.tar.gz

# /root/.cache を /code/.cache/ へのシンボリックリンクにする
RUN cd /root/ && ln -s /code/.cache/ .cache

# /root/.keras/ を /code/.cache/ へのシンボリックリンクにする
RUN cd /root/ && ln -s /code/.cache/ .keras

# /root/nltk_data/ を /code/.cache/ へのシンボリックリンクにする
RUN cd /root/ && ln -s /code/.cache/ nltk_data

# ソースコードをコピー
COPY ./ /code/

# ./Aivis.sh をエントリーポイントとして指定
ENTRYPOINT ["./Aivis.sh"]
