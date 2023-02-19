#!/usr/bin/env python3

# FutureWarning を抑制する
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Tensorflow のログを抑制する
## ref: https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import stable_whisper
import torch
import typer
from typing import cast

from Aivis import __version__
from Aivis import constants
from Aivis import demucs
from Aivis import prepare
from Aivis import utils


app = typer.Typer()

@app.command()
def segment(
    model_name: constants.ModelNameType = typer.Option('large-v2', help='Model name.'),
):
    typer.echo('=' * utils.GetTerminalColumnSize())

    # 01-Sources フォルダ以下のメディアファイルを取得
    ## 拡張子は .wav / .mp3 / .m4a / .mp4 / .ts
    ## アルファベット順にソートする
    source_files = sorted(list(constants.SOURCES_DIR.glob('**/*.*')))
    source_files = [i for i in source_files if i.suffix in constants.SOURCE_FILE_EXTENSIONS]

    # Demucs V4 (htdemucs_ft) で AI 音源分離を行い、音声ファイルからボイスのみを抽出する
    ## 本来は楽曲をボーカル・ドラム・ベース・その他に音源分離するための AI だが、これを応用して BGM・SE・ノイズなどを大部分除去できる
    ## Demucs でボーカル (=ボイス) のみを抽出したファイルは 02-PreparedSources/(音声ファイル名).wav に出力される
    ## すでに抽出済みのファイルがある場合は音源分離は行われず、すでに抽出済みのファイルを使用する
    voices_files = demucs.ExtractVoices(source_files, constants.PREPARE_SOURCES_DIR)

    # Whisper の推論を高速化し、メモリ使用率を下げて VRAM 8GB でも large-v2 モデルを使用できるようにする
    # ref: https://qiita.com/halhorn/items/d2672eee452ba5eb6241

    # Whisper の学習済みモデルをロード
    ## 一旦 CPU としてロードしてから CUDA に変更すると、メモリ使用量が大幅に削減されるらしい…
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
    _ = model.half()
    typer.echo('Run model.half() done.')
    typer.echo('Run model.cuda() ...')
    _ = model.cuda()
    typer.echo('Run model.cuda() done.')

    for voices_file in voices_files:
        typer.echo('=' * utils.GetTerminalColumnSize())

        # 出力先フォルダを作成
        ## すでに存在している場合は生成済みなのでスキップ (ただし、フォルダの中身が空の場合はスキップしない)
        ## もしもう一度生成したい場合はフォルダを削除すること
        folder = constants.SEGMENTS_DIR / voices_file.name.split('.')[0]
        if folder.exists() and len(list(folder.glob('*.*'))) > 0:
            typer.echo(f'Folder {folder} already exists. Skip.')
            continue
        folder.mkdir(parents=True, exist_ok=True)
        typer.echo(f'Folder {folder} created.')

        # すでに音声認識データ (JSON) がある場合はそのデータを使い、新規の音声認識は行わない
        finalized_results = []
        results_json_file = constants.PREPARE_SOURCES_DIR / f'{voices_file.name.split(".")[0]}.json'
        if results_json_file.exists():
            typer.echo(f'File {voices_file} already transcribed.')
            with open(results_json_file, mode='r', encoding='utf-8') as f:
                finalized_results = json.load(f)

        # 音声認識を開始
        else:
            typer.echo(f'File {voices_file} transcribing...')
            typer.echo('-' * utils.GetTerminalColumnSize())
            results = model.transcribe(
                str(voices_file),
                language = 'japanese',
                beam_size = 5,
                fp16 = True,
                verbose = True,
            )

            # 音声認識結果のタイムスタンプを調整し、ファイナライズする
            finalized_results = stable_whisper.finalize_segment_word_ts(results)
            finalized_results = [dict(text = ''.join(i), start = j[0]['start'], end = j[-1]['end']) for i, j in finalized_results]

            # 音声認識結果をファイルに保存
            with open(results_json_file, mode='w', encoding='utf-8') as f:
                json.dump(finalized_results, f, indent=4, ensure_ascii=False, allow_nan=True)

        # 一文ごとに切り出し、音声ファイルとその書き起こしのテキストファイルをペアで出力する
        count = 1
        for index, result in enumerate(finalized_results):
            typer.echo('-' * utils.GetTerminalColumnSize())

            # 出力先の音声ファイルと書き起こしのテキストファイルのパス
            output_audio_file = folder / f'Voices-{count:04d}.wav'
            output_transcription_file = folder / f'Voices-{count:04d}.txt'

            # 一文ごとに切り出した (セグメント化した) 音声ファイルを出力
            ## result['end'] を使うと文の末尾が切れることがあるため、次の文の開始位置を使う
            ## 一旦 start ~ end_max 間で余分に切り出してから、無音区間検出を行い、末尾の無音区間を削除する
            try:
                end_max = cast(float, finalized_results[index + 1]['start'])
            except IndexError:
                end_max = prepare.GetAudioFileDuration(voices_file)
            prepare.SliceAudioFile(voices_file, output_audio_file, cast(float, result['start']), cast(float, result['end']), end_max)

            # 出力した音声ファイルの長さが1秒未満になった場合、データセットにするには短すぎるためスキップする
            if prepare.GetAudioFileDuration(output_audio_file) < 1:
                output_audio_file.unlink()  # 出力した音声ファイルを削除
                typer.echo(f'File {output_audio_file} skipped. (Duration < 1 sec)')
                continue

            # 書き起こしのテキストファイルを出力
            with open(output_transcription_file, mode='w', encoding='utf-8') as f:
                f.write(prepare.PrepareText(result['text']))

            typer.echo(f'File {output_audio_file} saved.')
            count += 1

    typer.echo('=' * utils.GetTerminalColumnSize())
    typer.echo('All files segmentation done.')
    typer.echo('=' * utils.GetTerminalColumnSize())


@app.command()
def version():
    typer.echo(f'Aivis version {__version__}')


if __name__ == '__main__':
    app()
