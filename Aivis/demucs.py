
import typer
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from pydub import AudioSegment

from Aivis import utils


def ConvertToWave(file_paths: list[Path], output_dir: Path) -> list[Path]:
    """
    音声ファイルを WAV に変換して出力する

    Args:
        file_paths (list[Path]): ファイルパスのリスト
        output_dir (Path): 出力先のフォルダ

    Returns:
        list[Path]: 出力されたファイルパスのリスト
    """

    # 出力されたファイルパスのリスト (すでに変換済みのファイルも含む)
    output_file_paths: list[Path] = []

    for file_path in file_paths:
        typer.echo('=' * utils.GetTerminalColumnSize())

        # 出力先のファイルがすでに存在する場合
        # すでに変換済みなのでスキップ
        output_file_path = output_dir / f'{file_path.name.split(".")[0]}.wav'
        if output_file_path.exists():
            typer.echo(f'File {file_path} is already converted.')
            output_file_paths.append(output_file_path)
            continue

        typer.echo(f'File {file_path} converting...')
        typer.echo('-' * utils.GetTerminalColumnSize())

        # 音声ファイルを読み込む
        audio = AudioSegment.from_file(file_path)

        # 音声ファイルを WAV に変換する
        audio.export(output_file_path, format='wav')
        typer.echo('-' * utils.GetTerminalColumnSize())
        typer.echo(f'File {file_path} converted.')

        output_file_paths.append(output_file_path)
        typer.echo(f'File saved: {output_file_path}')

    return output_file_paths


def ExtractVoices(file_paths: list[Path], output_dir: Path) -> list[Path]:
    """
    音声ファイルからボイスのみを抽出 (BGM などは除去) して出力する

    Args:
        file_paths (list[Path]): ファイルパスのリスト
        output_dir (Path): 出力先のフォルダ

    Returns:
        list[Path]: 出力されたファイルパスのリスト
    """

    # Demucs での推論終了時に確実に VRAM を解放するため、マルチプロセスで実行する
    ## 確実に VRAM を解放できないと VRAM 容量次第では後続の Whisper での書き起こし処理に支障するため
    ## 並列処理は行っていない (リソース的に厳しい) ため本来はマルチプロセスにする意味はないが、
    ## マルチプロセスで起動させれば、マルチプロセス終了時に確実に VRAM を解放することができる
    ## del model でもある程度解放できるが、完全に解放されるわけではないみたい…
    ## 並列処理を行うためにマルチプロセスにしているわけではないため max_workers は 1 に設定している
    with ProcessPoolExecutor(max_workers=1) as executor:
        return executor.submit(__ExtractVoicesMultiProcess, file_paths, output_dir).result()


def __ExtractVoicesMultiProcess(file_paths: list[Path], output_dir: Path) -> list[Path]:
    """
    ProcessPoolExecutor で実行される ExtractVoices() の実処理

    Args:
        file_paths (list[Path]): ファイルパスのリスト
        output_dir (Path): 出力先のフォルダ

    Returns:
        list[Path]: 出力されたファイルパスのリスト
    """

    import torch
    from demucs.pretrained import get_model_from_args
    from demucs.repo import AnyModel
    from stable_whisper.audio import demucs_audio

    demucs_model: AnyModel | None = None

    # 出力されたファイルパスのリスト (すでに抽出済みのファイルも含む)
    output_file_paths: list[Path] = []

    for file_path in file_paths:

        typer.echo('=' * utils.GetTerminalColumnSize())

        # 出力先のファイルがすでに存在する場合
        # すでに抽出済みなのでスキップ
        output_file_path = output_dir / f'{file_path.name.split(".")[0]}.wav'
        if output_file_path.exists():
            typer.echo(f'File {file_path} is already separated.')
            output_file_paths.append(output_file_path)
            continue

        typer.echo(f'File {file_path} separating...')
        typer.echo('-' * utils.GetTerminalColumnSize())

        # 学習済みモデルを読み込む (初回のみ)
        if demucs_model is None:
            typer.echo('Demucs model loading...')
            demucs_model = get_model_from_args(type('args', (object,), dict(name='htdemucs_ft', repo=None))).cpu().eval()
            typer.echo('Demucs model loaded.')
            typer.echo('-' * utils.GetTerminalColumnSize())

        # 音源分離を実行する
        # 戻り値として torch.Tensor が返るが、今のところ使っていない
        demucs_audio(
            str(file_path),
            model = demucs_model,
            save_path = str(output_file_path),
            device = 'cuda',
            verbose = True,
        )
        typer.echo('-' * utils.GetTerminalColumnSize())
        typer.echo(f'File {file_path} separated.')

        output_file_paths.append(output_file_path)
        typer.echo(f'File saved: {output_file_path}')

    # GPU の VRAM を解放する
    del demucs_model
    torch.cuda.empty_cache()

    return output_file_paths
