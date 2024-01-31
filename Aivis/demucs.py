
from __future__ import annotations

import ffmpeg
import numpy as np
import typer
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from pydub import AudioSegment
from typing import Any, TYPE_CHECKING

# ロード時間短縮のため型チェック時のみインポートする
if TYPE_CHECKING:
    from torch import Tensor
    from demucs.htdemucs import HTDemucs

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

    demucs_model = None

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
        RunDemucs(
            demucs_model,
            str(file_path),
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


def RunDemucs(
    model: HTDemucs,
    audio: Tensor | str,
    input_sr: int | None = None,
    output_sr: int | None = None,
    device: str | None = None,
    verbose: bool = True,
    track_name: str | None = None,
    save_path: str | None = None,
    **demucs_options: Any,
) -> Tensor:
    """
    Demucs で音源分離を実行する
    stable-ts v2.14.4 時点での demucs_audio() を若干変更の上移植したもの
    stable-ts v2.15.0 以降では Demucs の 4 トラックのうち単一トラックのみを処理するよう大幅に音源分離関連が変更されたが、
    その結果雑音除去性能がガタ落ちしていたため (出力後の音声にブーンと低いノイズが入る…) 、あえて古い実装を移植して利用している
    その分遅くはなるが、ノイズが入ることで学習時に支障が出たら元も子もない
    ref: https://github.com/jianfch/stable-ts/blob/f6d61c228d5a00f89637422537d36cd358e5b90d/stable_whisper/audio.py
    """

    import torch
    import torchaudio
    from demucs.apply import apply_model

    def load_audio(file: str | bytes, sr: int = 44100):
        if isinstance(file, bytes):
            inp, file = file, 'pipe:'
        else:
            inp = None
        try:
            # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
            out, _ = (
                ffmpeg.input(file, threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True, input=inp)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    if track_name:
        track_name = f'"{track_name}"'

    if isinstance(audio, (str, bytes)):
        if not track_name:
            track_name = f'"{audio}"'
        audio = torch.from_numpy(load_audio(audio, model.samplerate))
    elif input_sr != model.samplerate:
        if input_sr is None:
            raise ValueError('No [input_sr] specified for audio tensor.')
        audio = torchaudio.functional.resample(audio,
                                               orig_freq=input_sr,
                                               new_freq=model.samplerate)
    if not track_name:
        track_name = 'audio track'
    audio_dims = audio.dim()
    if audio_dims == 1:
        audio = audio[None, None].repeat_interleave(2, -2)
    else:
        if audio.shape[-2] == 1:
            audio = audio.repeat_interleave(2, -2)
        if audio_dims < 3:
            audio = audio[None]

    if 'mix' in demucs_options:
        audio = demucs_options.pop('mix')

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    vocals_idx = model.sources.index('vocals')
    if verbose:
        print(f'Isolating vocals from {track_name}')
    apply_kwarg = dict(
        model=model,
        mix=audio,
        device=device,
        split=True,
        overlap=.25,
        progress=verbose is not None,
    )
    apply_kwarg.update(demucs_options)
    vocals = apply_model(**apply_kwarg)[0, vocals_idx].mean(0)  # type: ignore

    if device != 'cpu':
        torch.cuda.empty_cache()

    if output_sr is not None and model.samplerate != output_sr:
        vocals = torchaudio.functional.resample(vocals,
                                                orig_freq=model.samplerate,
                                                new_freq=output_sr)

    if save_path is not None:
        if not save_path.lower().endswith('.wav'):
            save_path += '.wav'
        torchaudio.save(save_path, vocals[None], output_sr or model.samplerate)  # type: ignore
        print(f'Saved: {save_path}')

    return vocals
