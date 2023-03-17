
import typer
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from Aivis import constants
from Aivis import utils


def Transcribe(model_name: constants.ModelNameType, voices_file: Path) -> list[dict[str, Any]]:
    """
    Whisper で音声認識を行う

    Args:
        model_name (constants.ModelNameType): 使用するモデルの名前 (基本は large-v2)
        voices_file (Path): 音声認識を行う音声ファイルのパス

    Returns:
        list[dict[str, Any]]: 音声認識結果のリスト
    """

    # Whisper での推論終了時に確実に VRAM を解放するため、マルチプロセスで実行する
    ## 終了を待機するため本来はマルチプロセスにする意味はないが、異なるプロセスにすることで
    ## プロセス終了時に確実に VRAM を解放することができる
    ## del model でもある程度解放できるが、完全に解放されるわけではないみたい…
    with ProcessPoolExecutor(max_workers=1) as executor:
        return executor.submit(__TranscribeMultiProcess, model_name, voices_file).result()


def __TranscribeMultiProcess(model_name: constants.ModelNameType, voices_file: Path) -> list[dict[str, Any]]:
    """
    ProcessPoolExecutor で実行される Transcribe() の実処理

    Args:
        model_name (constants.ModelNameType): 使用するモデルの名前 (基本は large-v2)
        voices_file (Path): 音声認識を行う音声ファイルのパス

    Returns:
        list[dict[str, Any]]: 音声認識結果のリスト
    """

    import stable_whisper
    import torch

    # Whisper の推論を高速化し、メモリ使用率を下げて VRAM 8GB でも large-v2 モデルを使用できるようにする
    ## 学習済みモデルのロードには時間がかかるので、本来であれば一度ロードしたモデルを再利用するべきだが、
    ## そうすると inaSpeechSegmenter の実行時に VRAM 8GB な GPU だと Out of memory になってしまうため、
    ## やむを得ず毎回モデルをロードしている
    ## ref: https://qiita.com/halhorn/items/d2672eee452ba5eb6241

    # Whisper の学習済みモデルをロード
    ## 一旦 CPU としてロードしてから CUDA に変更すると、メモリ使用量が大幅に削減されるらしい…
    typer.echo('-' * utils.GetTerminalColumnSize())
    typer.echo('Whisper model loading...')
    model = stable_whisper.load_model(model_name.value, device='cpu')
    typer.echo('Whisper model loaded.')

    # 学習済みモデルを JIT にコンパイル
    typer.echo('Whisper model compiling...')
    model.encoder = torch.jit.script(model.encoder)  # type: ignore
    model.decoder = torch.jit.script(model.decoder)  # type: ignore
    typer.echo('Whisper model compiled.')

    # 推論の高速化 & メモリ使用量の削減
    typer.echo('Run model.half() ...')
    model.half()
    typer.echo('Run model.half() done.')
    typer.echo('Run model.cuda() ...')
    model.cuda()
    typer.echo('Run model.cuda() done.')

    # 音声認識を開始
    typer.echo('-' * utils.GetTerminalColumnSize())
    typer.echo(f'File {voices_file} transcribing...')
    typer.echo('-' * utils.GetTerminalColumnSize())
    results = model.transcribe(
        str(voices_file),
        language = 'japanese',
        beam_size = 5,
        fp16 = True,
        verbose = True,
        # initial_prompt (呪文) を指定することで、書き起こし後のテキストのスタイルをある程度制御できるらしい…？
        ## 書き起こしに句読点が含まれやすくなるように調整しているが、付かないこともある…
        ## ref: https://github.com/openai/whisper/discussions/194
        ## ref: https://github.com/openai/whisper/discussions/204
        ## ref: https://github.com/openai/whisper/discussions/328
        initial_prompt = '次の文章は、アニメの物語の書き起こしです。必ず句読点をつけてください。',
    )
    typer.echo('-' * utils.GetTerminalColumnSize())
    typer.echo(f'File {voices_file} transcribed.')

    # 音声認識結果のタイムスタンプを調整し、ファイナライズする
    finalized_results_tmp = stable_whisper.finalize_segment_word_ts(results)
    finalized_results = [dict(text = ''.join(i), start = j[0]['start'], end = j[-1]['end']) for i, j in finalized_results_tmp]

    # GPU のメモリを解放する
    ## ref: https://github.com/openai/whisper/discussions/605
    del model.encoder
    del model.decoder
    del model
    torch.cuda.empty_cache()

    return finalized_results
