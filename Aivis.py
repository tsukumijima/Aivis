#!/usr/bin/env python3

# FutureWarning と RuntimeWarning を抑制する
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# Tensorflow のログを抑制する
## ref: https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import typer
from typing import Any, cast

from Aivis import __version__
from Aivis import constants
from Aivis import demucs
from Aivis import prepare
from Aivis import utils
from Aivis import whisper


# TODO: （）や【】「」で囲われたトランスクリプトが出る場合があるので、それを除外する
# それらで囲われた部分を除去してから、トランスクリプトが全滅した場合にスキップとか
# TODO: 後のセグメントの開始時刻が前のセグメントの終了時刻よりも前にならないようにする
# TODO: 開始時刻と終了時刻が同じ書き起こしはだいたいおかしいのでスキップ
# TODO: 句読点が連続しないようにする

app = typer.Typer()

@app.command()
def segment(
    model_name: constants.ModelNameType = typer.Option('large-v2', help='Model name.'),
    force_transcribe: bool = typer.Option(False, help='Force Whisper to transcribe audio files.'),
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

        finalized_results: list[dict[str, Any]] = []
        results_json_file = constants.PREPARE_SOURCES_DIR / f'{voices_file.name.split(".")[0]}.json'

        # すでに音声認識データ (JSON) がある場合はそのデータを使い、新規の音声認識は行わない
        ## なお、--force-transcribe オプションが指定されている場合は JSON ファイルが存在するかに関わらず音声認識を実行する
        if results_json_file.exists() and force_transcribe is False:
            typer.echo(f'File {voices_file} already transcribed.')
            with open(results_json_file, mode='r', encoding='utf-8') as f:
                finalized_results = json.load(f)

        # Whisper で音声認識を実行
        else:

            # 音声認識を実行し、タイムスタンプなどが調整された音声認識結果を取得する
            finalized_results = whisper.Transcribe(model_name, voices_file)

            # 音声認識結果をファイルに出力する
            with open(results_json_file, mode='w', encoding='utf-8') as f:
                json.dump(finalized_results, f, indent=4, ensure_ascii=False, allow_nan=True)

        # 一文ごとに切り出し、音声ファイルとその書き起こしのテキストファイルをペアで出力する
        count = 1
        for index, result in enumerate(finalized_results):
            typer.echo('-' * utils.GetTerminalColumnSize())

            # 出力先の音声ファイルと書き起こしのテキストファイルのパス
            output_audio_file = folder / f'Voices-{count:04d}.wav'
            output_transcription_file = folder / f'Voices-{count:04d}.txt'

            # 書き起こし結果を下処理し、よりデータセットとして最適な形にする
            transcription = prepare.PrepareText(result['text'])
            typer.echo(f'File {output_audio_file} Transcription: {transcription}')

            # (句読点含めて) 書き起こし結果が4文字未満だった場合、データセットにするには短すぎるためスキップする
            ## 例: そう。
            if len(transcription) < 4:
                typer.echo(f'File {output_audio_file} skipped. (Transcription length < 4 characters)')
                continue

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
                f.write(transcription)

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
