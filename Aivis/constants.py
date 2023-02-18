
from enum import Enum
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

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
