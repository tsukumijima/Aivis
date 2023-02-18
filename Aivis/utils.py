
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
