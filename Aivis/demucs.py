
# https://github.com/facebookresearch/demucs/blob/main/demucs/separate.py をベースに改変したもの

import torch
import typer
from demucs.apply import apply_model
from demucs.audio import save_audio
from demucs.pretrained import get_model
from demucs.repo import ModelLoadingError
from demucs.separate import load_track
from dora.log import fatal
from pathlib import Path

from Aivis import utils


def ExtractVoices(file_paths: list[Path], output_dir: Path) -> list[Path]:
    """
    音声ファイルからボイスのみを抽出 (BGM などは除去) して保存する

    Args:
        file_paths (list[Path]): ファイルパスのリスト
        output_dir (Path): 出力先のフォルダ

    Returns:
        list[Path]: 出力されたファイルパスのリスト
    """

    # 学習済みモデルを読み込む
    typer.echo('Demucs model loading...')
    try:
        model = get_model('htdemucs_ft')
    except ModelLoadingError as error:
        fatal(error.args[0])
    typer.echo('Demucs model loaded.')

    model.cpu()
    model.eval()

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

        # 音声ファイルを読み込む
        typer.echo(f'File {file_path} separating...')
        wav = load_track(file_path, model.audio_channels, model.samplerate)

        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()

        # 音源分離を実行する
        sources = apply_model(
            model,
            wav[None],
            device = 'cuda',
            shifts = 1,
            split = True,
            overlap = 0.25,
            progress = True,
            num_workers = 20,  # 20 ワーカーで並列処理する (GTX1080 なら余裕でいけた)
        )[0]  # type: ignore
        sources = sources * ref.std() + ref.mean()
        typer.echo(f'File {file_path} separated.')

        kwargs = {
            'samplerate': model.samplerate,
            'bitrate': 320,
            'clip': 'rescale',
            'as_float': False,
            'bits_per_sample': 16,
        }

        for source, name in zip(sources, model.sources):

            # Demucs はほかにもドラムやベース、それ以外などに分離/抽出してくれるが、ここでは vocals のみを保存する
            if name != 'vocals':
                continue

            save_audio(source, str(output_file_path), **kwargs)
            output_file_paths.append(output_file_path)
            typer.echo(f'File saved: {output_file_path}')

    typer.echo('=' * utils.GetTerminalColumnSize())

    # 今後の処理で別の AI を実行できるよう、GPU の VRAM を解放する
    del model
    torch.cuda.empty_cache()

    return output_file_paths
