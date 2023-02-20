
import os


def GetTerminalColumnSize() -> int:
    """
    ターミナルの列のサイズを取得する

    Returns:
        int: ターミナルの列のサイズ
    """

    try:
        columns = os.get_terminal_size().columns
        return columns
    except OSError:
        return 80


def SecondToTimeCode(second: float) -> str:
    """
    秒数をタイムコード (HH:MM:SS.mmm) に変換する

    Args:
        second (float): 秒数

    Returns:
        str: タイムコード
    """

    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    return f'{int(h):02d}:{int(m):02d}:{int(s):02d}.{int((s - int(s)) * 1000):03d}'
