
# CUDA 11.7 (Runtime Ubuntu 20.04) をベースイメージとして利用
FROM nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu20.04

# タイムゾーンを東京に設定
ENV TZ=Asia/Tokyo

# apt-get に対話的に設定を確認されないための設定
ENV DEBIAN_FRONTEND=noninteractive

# Python 3.10 と FFmpeg のインストール
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git software-properties-common tzdata && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        curl \
        python3.10 \
        python3.10-distutils \
        python3.10-venv \
        ffmpeg && \
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

# ソースコードをコピー
COPY ./ /code/

# Aivis.py をエントリーポイントとして指定
ENTRYPOINT ["pipenv", "run", "python", "Aivis.py"]
