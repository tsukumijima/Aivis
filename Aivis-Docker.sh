#!/bin/bash

# このファイルがあるディレクトリを取得
BASE_DIR=$(cd $(dirname $0); pwd)

# 第一引数に build が指定されている場合は Docker イメージを作成して終了する
if [ "$1" = "build" ]; then
    docker build -t aivis .
    exit 0
fi

# TODO: Docker イメージが安定しないうちは毎回ビルドする
docker build -t aivis .

# まだ Docker イメージがビルドされていない場合はビルドする
if [ ! "$(docker images -q aivis:latest 2> /dev/null)" ]; then
    docker build -t aivis .
fi

# Docker コンテナを起動する
## --gpus all で NVIDIA GPU をコンテナ内で使えるようにする
## データフォルダをコンテナ内にマウントする
## /code/.cache をマウントし、毎回学習済みモデルがダウンロードされるのを防ぐ
## --shm-size を指定しないと DataLoader でエラーが発生する
## ref: https://qiita.com/gorogoroyasu/items/e71dd3c076af145c9b44
docker run --gpus all -it --rm --shm-size=256m \
    -p 7860:7860 \
    -v ${BASE_DIR}/.cache:/code/.cache \
    -v ${BASE_DIR}/01-Sources:/code/01-Sources \
    -v ${BASE_DIR}/02-PreparedSources:/code/02-PreparedSources \
    -v ${BASE_DIR}/03-Segments:/code/03-Segments \
    -v ${BASE_DIR}/04-Datasets:/code/04-Datasets \
    -v ${BASE_DIR}/Bert-VITS2:/code/Bert-VITS2 \
    aivis "$@"
