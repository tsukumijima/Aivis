#!/bin/bash

# faster-whisper が動的にシステム上の libcudnn_ops_infer.so.8 をロードしようとしてエラーになるので、
# 事前に .venv/ 以下のライブラリへ LD_LIBRARY_PATH を通しておく
# ref: https://github.com/SYSTRAN/faster-whisper/issues/153#issuecomment-1510218906
LD_LIBRARY_PATH=`poetry run python -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'` poetry run python -m Aivis "$@"
