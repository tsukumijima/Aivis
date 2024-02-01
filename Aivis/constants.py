
from enum import Enum
from pathlib import Path


# 各データの保存先ディレクトリ
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / '.cache'
SOURCES_DIR = BASE_DIR / '01-Sources'
PREPARE_SOURCES_DIR = BASE_DIR / '02-PreparedSources'
SEGMENTS_DIR = BASE_DIR / '03-Segments'
DATASETS_DIR = BASE_DIR / '04-Datasets'
BERT_VITS2_DIR = BASE_DIR / 'Bert-VITS2'

# データソースとして読み込むファイルの拡張子
## 大半の動画・音声ファイルを網羅しているはず
SOURCE_FILE_EXTENSIONS = [
    '.wav',
    '.flac',
    '.opus',
    '.ogg',
    '.vorbis',
    '.mp3',
    '.m4a',
    '.mp4',
    '.mov',
    '.mkv',
    '.webm',
    '.wmv',
    '.ts',
    '.mts',
    '.m2ts',
    '.mpg',
    '.mpeg',
]

# スキップする Whisper のハルシネーション避けのワード
SKIP_TRANSCRIPTS = [
    '視聴ありがとう',
    '視聴頂き',
    '視聴いただき',
    '視聴下さ',
    '視聴くださ',
    'チャンネル登録',
]

class ModelNameType(str, Enum):
    small = 'small'
    medium = 'medium'
    large = 'large'
    large_v1 = 'large-v1'
    large_v2 = 'large-v2'
    large_v3 = 'large-v3'
