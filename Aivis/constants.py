
from enum import Enum
from pathlib import Path


# 各データの保存先ディレクトリ
BASE_DIR = Path(__file__).resolve().parent.parent
SOURCES_DIR = BASE_DIR / '01-Sources'
PREPARE_SOURCES_DIR = BASE_DIR / '02-PrepareSources'
SEGMENTS_DIR = BASE_DIR / '03-Segments'
DATASETS_DIR = BASE_DIR / '04-Datasets'

# データソースとして読み込むファイルの拡張子
SOURCE_FILE_EXTENSIONS = [
    '.wav',
    '.mp3',
    '.m4a',
    '.mp4',
    '.ts',
]

class DeviceType(str, Enum):
    cpu = 'cpu'
    cuda = 'cuda'

class ModelNameType(str, Enum):
    small = 'small'
    medium = 'medium'
    large = 'large'
    large_v1 = 'large-v1'
    large_v2 = 'large-v2'
