#!/bin/bash

# このファイルがあるディレクトリを取得
BASE_DIR=$(cd $(dirname $0); pwd)

# まだ Docker イメージが作成されていない場合は作成する
if [ ! "$(docker images -q aivis:latest 2> /dev/null)" ]; then
    docker build -t aivis .
fi

# Docker コンテナを起動する
## --gpus all で NVIDIA GPU をコンテナ内で使えるようにする
## データフォルダをコンテナ内にマウントする
docker run --gpus all -it --rm \
    -v ${BASE_DIR}/01-Sources:/code/01-Sources \
    -v ${BASE_DIR}/02-PrepareSources:/code/02-PrepareSources \
    -v ${BASE_DIR}/03-Segments:/code/03-Segments \
    -v ${BASE_DIR}/04-Datasets:/code/04-Datasets aivis $@
