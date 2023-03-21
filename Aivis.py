#!/usr/bin/env python3

# FutureWarning・RuntimeWarning・UserWarning を抑制する
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import json
import stable_whisper
import typer
import whisper
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
    force_transcribe: bool = typer.Option(False, help='Force Whisper to transcribe audio files.'),
):

    # 01-Sources フォルダ以下のメディアファイルを取得
    ## 拡張子は .wav / .mp3 / .m4a / .mp4 / .ts
    ## アルファベット順にソートする
    source_files = sorted(list(constants.SOURCES_DIR.glob('**/*.*')))
    source_files = [i for i in source_files if i.suffix in constants.SOURCE_FILE_EXTENSIONS]

    # Demucs V4 (htdemucs) で AI 音源分離を行い、音声ファイルからボイスのみを抽出する
    ## 本来は楽曲をボーカル・ドラム・ベース・その他に音源分離するための AI だが、これを応用して BGM・SE・ノイズなどを大部分除去できる
    ## Demucs でボーカル (=ボイス) のみを抽出したファイルは 02-PreparedSources/(音声ファイル名).wav に出力される
    ## すでに抽出済みのファイルがある場合は音源分離は行われず、すでに抽出済みのファイルを使用する
    voices_files = demucs.ExtractVoices(source_files, constants.PREPARE_SOURCES_DIR)

    model: whisper.Whisper | None = None

    # ここからは各音声ファイルごとにループ
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

        result: stable_whisper.WhisperResult
        results_json_file = constants.PREPARE_SOURCES_DIR / f'{voices_file.name.split(".")[0]}.json'

        # すでに音声認識結果のデータ (JSON) が保存されている場合はそのデータを使い、新規の音声認識は行わない
        ## なお、--force-transcribe オプションが指定されている場合は JSON ファイルが存在するかに関わらず音声認識を実行する
        if results_json_file.exists() and force_transcribe is False:
            typer.echo(f'File {voices_file} already transcribed.')
            result = stable_whisper.WhisperResult(str(results_json_file))

        # Whisper で音声認識を実行
        else:

            # Whisper の学習済みモデルをロード (1回のみ)
            if model is None:
                typer.echo('-' * utils.GetTerminalColumnSize())
                typer.echo('Whisper model loading...')
                model = stable_whisper.load_model(
                    model_name.value,
                    device = 'cuda',
                    cpu_preload = True,
                    dq = False,
                )
                typer.echo('Whisper model loaded.')

            # Whisper に入力する初期プロンプト (呪文)
            ## Whisper は前の文脈を踏まえて書き起こしてくれるらしいので、書き起こしっぽいものを入れておくと、
            ## 書き起こしに句読点をつけるよう誘導できるみたい…
            initial_prompt = (
                'そうだ。今日はピクニックしない？天気もいいし、絶好のピクニック日和だと思う。いいですね。'
                'では、準備をはじめましょうか。そうしよう！どこに行く？そうですね。三ツ池公園なんか良いんじゃないかな。'
                '今の時期なら桜が綺麗だしね。じゃあそれで決まり！わかりました。電車だと550円掛かるみたいです。'
                '少し時間が掛かりますが、歩いた方が健康的かもしれません。'
            )

            # 音声認識を実行し、タイムスタンプなどが調整された音声認識結果を取得する
            typer.echo('-' * utils.GetTerminalColumnSize())
            typer.echo(f'File {voices_file} transcribing...')
            typer.echo('-' * utils.GetTerminalColumnSize())
            result = cast(stable_whisper.WhisperResult, model.transcribe(
                # 入力元の音声ファイル
                str(voices_file),
                # ログをコンソールに出力する
                verbose = True,
                # 初期プロンプト
                initial_prompt = initial_prompt,
                # 単語セグメントの再グループ化を行わない
                ## 日本語の場合、再グループ化を行うと、話者を跨いで同じ一区切りにされてしまうことがあるなど
                ## 今回の用途には適さないため、再グループ化を行わないようにしている
                regroup = False,
                # すでに Demucs で音源分離を行っているため、ここでは音源分離を行わない
                ## 音声ファイルごとにモデルを読み込むよりも、読み込んだモデルを使いまわした方が高速に処理できる
                demucs = False,
                # Silero VAD を使用してタイムスタンプ抑制マスクを生成する
                vad = True,
                # Whisper 本体の設定パラメータ (whisper.decoding.DecodingOptions)
                language = 'Japanese',  # 日本語
                temperature = (0.0, 0.2, 0.4, 0.6),  # 低い値にしてランダム性を抑える
                suppress_tokens = '-1',
                fp16 = True,
            ))
            typer.echo('-' * utils.GetTerminalColumnSize())
            typer.echo(f'File {voices_file} transcribed.')

            # 音声認識結果をファイルに出力する
            with open(results_json_file, mode='w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=4, ensure_ascii=False, allow_nan=True)

        # 一文ごとに切り出した音声ファイル（ファイル名には書き起こし文が入る）を出力する
        count = 1
        for segment in result.segments:
            typer.echo('-' * utils.GetTerminalColumnSize())

            # 書き起こし結果を下処理し、よりデータセットとして最適な形にする
            transcription = prepare.PrepareText(segment.text)
            typer.echo(f'Transcription: {transcription}')

            # (句読点含めて) 書き起こし結果が4文字未満だった場合、データセットにするには短すぎるためスキップする
            ## 例: そう。
            if len(transcription) < 4:
                typer.echo(f'Transcription skipped. (Transcription length < 4 characters)')
                continue

            # 開始時刻と終了時刻が同じだった場合、タイムスタンプが正しく取得できていないためスキップする
            if segment.start == segment.end:
                typer.echo(f'Transcription skipped. (Start time == End time)')
                continue

            # 出力先の音声ファイルのパス
            # 例: 0001_こんにちは.wav
            output_audio_file = folder / f'{count:04d}_{transcription}.wav'

            # 一文ごとに切り出した (セグメント化した) 音声ファイルを出力
            typer.echo(f'Segment Range: {utils.SecondToTimeCode(segment.start)} - {utils.SecondToTimeCode(segment.end)}')
            prepare.SliceAudioFile(voices_file, output_audio_file, segment.start, segment.end)

            # 出力した音声ファイルの長さが1秒未満になった場合、データセットにするには短すぎるためスキップする
            if prepare.GetAudioFileDuration(output_audio_file) < 1:
                output_audio_file.unlink()  # 出力した音声ファイルを削除
                typer.echo(f'Transcription skipped. (Duration < 1 sec)')
                continue

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
