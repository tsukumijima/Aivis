#!/usr/bin/env python3

# FutureWarning / RuntimeWarning / UserWarning を抑制する
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import functools
import json
import math
import re
import shutil
import subprocess
import sys
import typer
from pathlib import Path
from typing import Annotated, Any, cast, Optional, Union

from Aivis import __version__
from Aivis import constants
from Aivis import demucs
from Aivis import prepare
from Aivis import utils


app = typer.Typer(help='Aivis: AI Voice Imitation System')

@app.command(help='Create audio segments from audio sources.')
def create_segments(
    use_demucs: Annotated[bool, typer.Option(help='Use Demucs to extract voices from audio files.')] = True,
    whisper_model: Annotated[constants.ModelNameType, typer.Option(help='Whisper model name.')] = constants.ModelNameType.large_v3,
    force_transcribe: Annotated[bool, typer.Option(help='Force Whisper to transcribe audio files.')] = False,
    trim_silence: Annotated[bool, typer.Option(help='Trim silence (start and end only) from audio files.')] = True,
):
    # このサブコマンドでしか利用せず、かつ比較的インポートが重いモジュールはここでインポートする
    import faster_whisper
    import stable_whisper

    # 01-Sources ディレクトリ以下のメディアファイルを取得
    ## 処理対象のメディアファイルの拡張子は constants.SOURCE_FILE_EXTENSIONS で定義されている
    ## アルファベット順にソートする
    source_files = sorted(list(constants.SOURCES_DIR.glob('**/*.*')))
    source_files = [i for i in source_files if i.suffix in constants.SOURCE_FILE_EXTENSIONS]

    # Demucs V4 (htdemucs_ft) で AI 音源分離を行い、音声ファイルからボイスのみを抽出する
    ## 本来は楽曲をボーカル・ドラム・ベース・その他に音源分離するための AI だが、これを応用して BGM・SE・ノイズなどを大部分除去できる
    ## Demucs でボーカル (=ボイス) のみを抽出したファイルは 02-PreparedSources/(音声ファイル名).wav に出力される
    ## すでに抽出済みのファイルがある場合は音源分離は行われず、すでに抽出済みのファイルを使用する
    ## Demucs での音源分離を行わない場合は、音声ファイルを wav に変換して 02-PreparedSources/(音声ファイル名).wav に出力する
    if use_demucs is True:
        voices_files = demucs.ExtractVoices(source_files, constants.PREPARE_SOURCES_DIR)
    else:
        voices_files = demucs.ConvertToWave(source_files, constants.PREPARE_SOURCES_DIR)

    model: faster_whisper.WhisperModel | None = None

    # ここからは各音声ファイルごとにループ
    for voices_file in voices_files:
        typer.echo('=' * utils.GetTerminalColumnSize())

        # 出力先ディレクトリを作成
        ## すでに存在している場合は生成済みなのでスキップ (ただし、ディレクトリの中身が空の場合はスキップしない)
        ## もしもう一度生成したい場合はディレクトリを削除すること
        folder = constants.SEGMENTS_DIR / voices_file.name.split('.')[0]
        if folder.exists() and len(list(folder.glob('*.*'))) > 0:
            typer.echo(f'Directory {folder} already exists. Skip.')
            continue
        folder.mkdir(parents=True, exist_ok=True)
        typer.echo(f'Directory {folder} created.')

        transcribe_result: stable_whisper.WhisperResult
        results_json_file = constants.PREPARE_SOURCES_DIR / f'{voices_file.name.split(".")[0]}.json'

        # すでに音声認識結果のデータ (JSON) が保存されている場合はそのデータを使い、新規の音声認識は行わない
        ## なお、--force-transcribe オプションが指定されている場合は JSON ファイルが存在するかに関わらず音声認識を実行する
        if results_json_file.exists() and force_transcribe is False:
            typer.echo(f'File {voices_file} already transcribed.')
            transcribe_result = stable_whisper.WhisperResult(str(results_json_file))

        # Whisper で音声認識を実行
        else:

            typer.echo('-' * utils.GetTerminalColumnSize())
            typer.echo(f'File {voices_file} transcribing...')
            typer.echo('-' * utils.GetTerminalColumnSize())

            # Whisper の学習済みモデルをロード (1回のみ)
            if model is None:
                typer.echo(f'Whisper model loading... (Model: {whisper_model.value})')
                model = stable_whisper.load_faster_whisper(
                    whisper_model.value,
                    device = 'cuda',
                    compute_type = 'auto',
                )
                typer.echo('Whisper model loaded.')
                typer.echo('-' * utils.GetTerminalColumnSize())

            # Whisper に入力する初期プロンプト (呪文)
            ## Whisper は前の文脈を踏まえて書き起こしてくれるらしいので、会話文の書き起こしっぽいものを入れておくと、
            ## 書き起こしに句読点をつけるよう誘導できるみたい…
            initial_prompt = (
                'そうだ。今日はピクニックしない…？天気もいいし、絶好のピクニック日和だと思う！ 良いですね！行きましょう…！'
                'じゃあ早速、荷物の準備しておきますね。 そうだね！どこに行く？ そうですね…。桜の見える公園なんかどうでしょう…？'
                'おー！今の時期は桜が綺麗だしね。じゃあそれで決まりっ！ 分かりました。調べたところ、電車だと550円掛かるみたいです。'
                '少し時間が掛かりますが、歩いた方が健康的かもしれません。 え〜！歩くのはきついよぉ…。'
            )

            # 音声認識を実行し、タイムスタンプなどが調整された音声認識結果を取得する
            # ref: https://qiita.com/reriiasu/items/5ad8e1a7dbc425de7bb0
            # ref: https://zenn.dev/tsuzukia/articles/1381e6c9a88577
            # ref: https://note.com/asahi_ictrad/n/nf3ca329f17df
            transcribe_result: stable_whisper.WhisperResult = cast(Any, model).transcribe_stable(
                # 入力元の音声ファイル
                str(voices_file),
                # 単語ごとのタイムスタンプを出力する
                word_timestamps = True,
                # ログをコンソールに出力する
                verbose = True,
                # 単語セグメントの再グループ化を行わない
                ## 別途音声認識が完了してから行う
                regroup = False,
                # すでに Demucs で音源分離を行っているため、ここでは音源分離を行わない
                ## 音声ファイルごとにモデルを読み込むよりも、読み込んだモデルを使いまわした方が高速に処理できる
                demucs = False,
                # 検出された無音に基づいてタイムスタンプの調整を有効にする
                suppress_silence = True,
                # 検出された無音に基づいて単語のタイムスタンプを調整する
                suppress_word_ts = True,
                # Silero VAD を使用してタイムスタンプ抑制マスクを生成する
                vad = True,
                # faster-whisper 本体の設定パラメータ
                # 日本語
                language = 'ja',
                # beam_size (1 に設定して CER を下げる)
                beam_size = 1,
                # 謎のパラメータ (10 に設定すると temperature を下げたことで上がる repetition を抑えられるらしい？)
                no_repeat_ngram_size = 10,
                # temperature (0.0 に設定して CER を下げる)
                temperature = 0.0,
                # 前回の音声チャンクの出力結果を次のウインドウのプロンプトに設定しない
                condition_on_previous_text = False,
                # 初期プロンプト
                initial_prompt = initial_prompt,
                # faster-whisper 側で VAD を使った無音フィルタリングを行う
                vad_filter = True,
            )
            typer.echo('-' * utils.GetTerminalColumnSize())
            typer.echo(f'File {voices_file} transcribed.')

            # 音声認識結果を再グループ化する
            ## 再グループ化のアルゴリズムは多くあるが、ここではデフォルト設定を調整して使っている
            ## ref: https://github.com/jianfch/stable-ts#regrouping-words
            (transcribe_result.clamp_max()
                .split_by_punctuation([('.', ' '), '。', '?', '？', (',', ' '), '，'])  # type: ignore
                .split_by_gap(0.75)
                .merge_by_gap(0.3, max_words=3)
                .split_by_punctuation([('.', ' '), '。', '?', '？']))  # type: ignore

            # 音声認識結果をファイルに出力する
            with open(results_json_file, mode='w', encoding='utf-8') as f:
                json.dump(transcribe_result.to_dict(), f, indent=4, ensure_ascii=False, allow_nan=True)

        # 一文ごとに切り出した音声ファイル（ファイル名には書き起こし文が入る）を出力する
        count = 1
        for index, segment in enumerate(transcribe_result.segments):
            typer.echo('-' * utils.GetTerminalColumnSize())

            # 書き起こし結果を下処理し、よりデータセットとして最適な形にする
            transcript = prepare.PrepareText(segment.text)
            typer.echo(f'Transcript: {transcript}')

            # Whisper は無音区間とかがあると「視聴頂きありがとうございました」「チャンネル登録よろしく」などの謎のハルシネーションが発生するので、
            # そういう系の書き起こし結果があった場合はスキップする
            if transcript in constants.SKIP_TRANSCRIPTS:
                typer.echo(f'Transcript skipped. (Transcript is in SKIP_TRANSCRIPTS)')
                continue

            # (句読点含めて) 書き起こし結果が4文字未満だった場合、データセットにするには短すぎるためスキップする
            ## 例: そう。/ まじ？ / あ。
            if len(transcript) < 4:
                typer.echo(f'Transcript skipped. (Transcript length < 4 characters)')
                continue

            # セグメントの開始時間と終了時間を取得
            segment_start = segment.start
            segment_end = segment.end

            # もし現在処理中のセグメントの最初の単語の長さが 0.425 秒以上だった場合、先頭 0.25 秒を削る
            ## 前のセグメントの最後の発音の母音が含まれてしまう問題の回避策
            ## 日本語の場合単語は基本1文字か2文字になるため、発声時間は 0.425 秒以下になることが多いのを利用している
            if segment.words[0].duration >= 0.425:
                segment_start += 0.25

                # さらに、もし現在処理中のセグメントの最初の単語の長さが 1 秒以上だった場合、
                # その長さ - 1 秒をさらに削る (最低でも 0.75 秒は残す)
                ## 例: 3.6 秒ある単語なら、先頭 0.25 秒 + 2.6 秒 = 先頭 2.85 秒を削り、残りの 0.75 秒を出力する
                ## 1単語の発声に 1 秒以上掛かることはほぼあり得ないため、無音区間が含まれていると判断する
                if segment.words[0].duration >= 1.0:
                    segment_start += segment.words[0].duration - 1.0

            # もし次のセグメントの最初の単語の長さが 0.425 秒以上だった場合、末尾 0.25 秒を伸ばす
            ## 最後の発音の母音が切れてしまう問題の回避策
            if index + 1 < len(transcribe_result.segments) and transcribe_result.segments[index + 1].words[0].duration >= 0.425:
                segment_end += 0.25

                # さらに、もし次のセグメントの最初の単語の長さが 1 秒以上だった場合、
                # その長さ - 1 秒をさらに伸ばす (最大で 1.0 秒まで伸ばす)
                if transcribe_result.segments[index + 1].words[0].duration >= 1.0:
                    segment_end += min(transcribe_result.segments[index + 1].words[0].duration - 1.0, 1.0)

            # もし次のセグメントの開始位置が現在処理中のセグメントの終了位置よりも後なら、
            # 現在処理中のセグメントの終了位置を次のセグメントの開始位置に合わせて末尾が欠けないようにする (最大で 3.0 秒まで伸ばす)
            if index + 1 < len(transcribe_result.segments) and segment_end < transcribe_result.segments[index + 1].start:
                segment_end = min(transcribe_result.segments[index + 1].start, segment_end + 3.0)

            # もし現在処理中のセグメントが音声認識結果の最後のセグメントなら、
            # 現在処理中のセグメントの終了位置を音声の長さに合わせて末尾が欠けないようにする
            if index + 1 == len(transcribe_result.segments):
                segment_end = prepare.GetAudioFileDuration(voices_file)

            typer.echo(f'Segment Range: {utils.SecondToTimeCode(segment_start)} - {utils.SecondToTimeCode(segment_end)}')

            # 開始時刻と終了時刻が同じだった場合、タイムスタンプが正しく取得できていないためスキップする
            if segment_start == segment_end:
                typer.echo(f'Transcript skipped. (Start time == End time)')
                continue

             # 出力する音声ファイルの長さが1秒未満になった場合、データセットにするには短すぎるためスキップする
            if segment_end - segment_start < 1:
                typer.echo(f'Transcript skipped. (Duration < 1 sec)')
                continue

            # 出力先の音声ファイルのパス
            # 例: 0001_こんにちは.wav
            output_audio_file = folder / f'{count:04d}_{transcript}.wav'

            # 一文ごとに切り出した (セグメント化した) 音声ファイルを出力
            real_output_audio_file = prepare.SliceAudioFile(voices_file, output_audio_file, segment_start, segment_end, trim_silence)

            typer.echo(f'File {real_output_audio_file} saved.')
            count += 1

    typer.echo('=' * utils.GetTerminalColumnSize())
    typer.echo('All files segmentation done.')
    typer.echo('=' * utils.GetTerminalColumnSize())


@app.command(help='Create datasets from audio segments.')
def create_datasets(
    segments_dir_name: Annotated[str, typer.Argument(help='Segments directory name. Glob pattern (wildcard) is available.')],
    speaker_names: Annotated[str, typer.Argument(help='Speaker name. (Comma separated)')],
    accept_all: Annotated[bool, typer.Option(help='Accept all segments and transcriptions. (Skip UI)')] = False,
):
    # このサブコマンドでしか利用せず、かつ比較的インポートが重いモジュールはここでインポートする
    import gradio
    from gradio import WaveformOptions

    typer.echo('=' * utils.GetTerminalColumnSize())

    # バリデーション
    if speaker_names == '':
        typer.echo(f'Error: Speaker names is empty.')
        typer.echo('=' * utils.GetTerminalColumnSize())
        sys.exit(1)

    # 出力後のデータセットの出力先ディレクトリがなければ作成
    speaker_name_list = speaker_names.split(',')
    for speaker in speaker_name_list:
        output_dir = constants.DATASETS_DIR / speaker
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            typer.echo(f'Speaker: {speaker} / Directory: {output_dir} created.')
        else:
            typer.echo(f'Speaker: {speaker} / Directory: {output_dir} already created.')
    typer.echo('=' * utils.GetTerminalColumnSize())

    # 03-Segments/(指定されたディレクトリ名の Glob パターン)/ 以下のセグメント化された音声ファイルを取得
    ## 拡張子は .wav
    ## glob() の結果は順序がバラバラなのでアルファベット順にソートする
    segment_audio_paths = sorted(list((constants.SEGMENTS_DIR).glob(f'{segments_dir_name}/*.wav')))
    if len(segment_audio_paths) == 0:
        typer.echo(f'Error: {segments_dir_name}/*.wav glob pattern matched no files.')
        typer.echo('=' * utils.GetTerminalColumnSize())
        sys.exit(1)
    for segment_audio_path in segment_audio_paths:
        segments_dir_name = segment_audio_path.parent.name
        typer.echo(f'Segment File: {segments_dir_name}/{segment_audio_path.name}')
    typer.echo('=' * utils.GetTerminalColumnSize())

    # 音声ファイル名から書き起こし文を取得
    ## 例: 0001_こんにちは.wav -> こんにちは
    segment_audio_transcripts: list[str] = []
    for segment_audio_path in segment_audio_paths:

        # 拡張子なしファイル名から _ より後の部分を取得
        segment_audio_transcript = segment_audio_path.stem.split('_')[1]

        # 1文が長すぎてファイル名が最大文字数を超えてしまっている場合、別途同じファイル名で .txt ファイルに全体の書き起こし文が保存されているので、
        # それを読み込んで使う
        segment_audio_transcript_txt = segment_audio_path.with_suffix('.txt')
        if segment_audio_transcript_txt.exists():
            with open(segment_audio_transcript_txt, mode='r', encoding='utf-8') as f:
                segment_audio_transcript = f.read()

        # 書き起こし文をリストに追加
        segment_audio_transcripts.append(segment_audio_transcript)

    # 現在処理中の音声ファイルのインデックスと音声ファイルのパスと書き起こし文
    current_index = 0

    # セレクトボックスの選択肢
    choices = speaker_name_list

    # 出力ファイルの連番
    output_audio_count: dict[str, int] = {}
    for speaker in speaker_name_list:
        # 既にそのディレクトリに存在するファイルの中で連番が一番大きいものを取得し、それに 1 を足したものを初期値とする
        output_audio_count[speaker] = max([
            int(re.sub(r'\D', '', i.stem)) for i in (constants.DATASETS_DIR / speaker / 'audios').glob('*.wav')
        ], default=0) + 1

    # --accept-all を指定して UI を表示せずにすべての音声ファイルを一括処理する場合
    if accept_all is True:

        # --accept-all を指定した場合、話者名は必ず1つだけでなければならない
        ## 当然ながら、--accept-all を使う際は処理対象に指定したすべてのセグメントが同一話者のものでなければならない
        if len(speaker_name_list) != 1:
            typer.echo(f'Error: Speaker names must be one if --accept-all option is specified.')
            typer.echo('=' * utils.GetTerminalColumnSize())
            sys.exit(1)
        speaker_name = speaker_name_list[0]

        # 現在処理中の音声ファイルのインデックスが音声ファイルの総数に達するまでループ
        while current_index < len(segment_audio_paths):

            segment_audio_path = segment_audio_paths[current_index]
            transcript = segment_audio_transcripts[current_index]
            typer.echo(f'Segment File : {segment_audio_path.name}')
            typer.echo(f'Speaker Name : {speaker_name}')
            typer.echo(f'Transcript   : {transcript}')

            # データセットに音声ファイルを保存 (書き起こし文はファイル名が長くなるので含まず、別途書き起こしファイルに保存する)
            audio_output_dir = constants.DATASETS_DIR / speaker_name / 'audios'
            audio_output_dir.mkdir(parents=True, exist_ok=True)
            output_path = audio_output_dir / f'{output_audio_count[speaker_name]:04}.wav'
            output_audio_count[speaker_name] += 1  # 連番をインクリメント
            shutil.copyfile(segment_audio_path, output_path)
            typer.echo(f'File {output_path} saved.')

            # 音声ファイルのパスと書き起こし文のパスのペアを transcripts.list に順次追記
            text_list_path = constants.DATASETS_DIR / speaker_name / 'transcripts.list'
            if not text_list_path.exists():  # ファイルがなければ空のファイルを作成
                text_list_path.parent.mkdir(parents=True, exist_ok=True)
                text_list_path.touch()
            with open(text_list_path, mode='a', encoding='utf-8') as f:
                f.write(f'{output_path.name}|{speaker_name}|JP|{transcript}\n')
            typer.echo(f'File {text_list_path} updated.')
            typer.echo('-' * utils.GetTerminalColumnSize())

            # 次の処理対象のファイルのインデックスに進める
            current_index += 1

        # すべての音声ファイルを処理したら終了
        typer.echo('=' * utils.GetTerminalColumnSize())
        typer.echo('All files processed.')
        typer.echo('=' * utils.GetTerminalColumnSize())
        return

    def OnClick(
        segment_audio_path_str: str,
        speaker_name: str,
        transcript: str,
        is_skip: bool = False,
    ) -> tuple[gradio.Audio, gradio.Dropdown, gradio.Textbox]:
        """ 確定ボタンが押されたときの処理 """

        nonlocal current_index, segment_audio_paths, segment_audio_transcripts, choices, output_audio_count

        # 話者名が空の場合は初期画面から「確定」を押して実行されたイベントなので、保存処理は実行しない
        speaker_name = speaker_name.strip()
        if speaker_name != '' and speaker_name != '選別完了':

            segment_audio_path = Path(segment_audio_path_str)
            typer.echo(f'Segment File : {segment_audio_path.name}')
            typer.echo(f'Speaker Name : {speaker_name if is_skip is False else "(Skipped)"}')
            typer.echo(f'Transcript   : {transcript}')

            # 確定ボタンの代わりにスキップボタンが押された場合は何もしない
            if is_skip is True:
                typer.echo('Segment file skipped.')
                typer.echo('-' * utils.GetTerminalColumnSize())
            else:
                # データセットに編集後の音声ファイルを保存 (書き起こし文はファイル名が長くなるので含まず、別途書き起こしファイルに保存する)
                ## Gradio の謎機能で、GUI でトリムした編集後の一次ファイルが segment_audio_path_str として渡されてくる
                audio_output_dir = constants.DATASETS_DIR / speaker_name / 'audios'
                audio_output_dir.mkdir(parents=True, exist_ok=True)
                output_path = audio_output_dir / f'{output_audio_count[speaker_name]:04}.wav'
                output_audio_count[speaker_name] += 1  # 連番をインクリメント
                shutil.copyfile(segment_audio_path, output_path)
                typer.echo(f'File {output_path} saved.')

                # 音声ファイルのパスと書き起こし文のパスのペアを transcripts.list に順次追記
                text_list_path = constants.DATASETS_DIR / speaker_name / 'transcripts.list'
                if not text_list_path.exists():  # ファイルがなければ空のファイルを作成
                    text_list_path.parent.mkdir(parents=True, exist_ok=True)
                    text_list_path.touch()
                with open(text_list_path, mode='a', encoding='utf-8') as f:
                    f.write(f'{output_path.name}|{speaker_name}|JP|{transcript}\n')
                typer.echo(f'File {text_list_path} updated.')
                typer.echo('-' * utils.GetTerminalColumnSize())

            # 次の処理対象のファイルのインデックスに進める
            current_index += 1

        elif current_index < len(segment_audio_paths):
            # 初期画面から「確定」を押して実行されたイベントなので、ログに確定を出力
            ## 次の処理対象のファイルがない場合は実行されない
            typer.echo('=' * utils.GetTerminalColumnSize())
            typer.echo('Selection of segment files has started.')
            typer.echo('=' * utils.GetTerminalColumnSize())

        # 次の処理対象のファイルがない場合は終了
        if current_index >= len(segment_audio_paths):
            typer.echo('=' * utils.GetTerminalColumnSize())
            typer.echo('All files processed.')
            typer.echo('=' * utils.GetTerminalColumnSize())
            return (
                gradio.Audio(
                    sources = [],
                    type = 'filepath',
                    interactive = True,
                    autoplay = True,
                    waveform_options = WaveformOptions(sample_rate=44100),  # UI 上でトリミングした音声ファイルのサンプリングレート
                ),
                gradio.Dropdown(choices=['選別完了'], value='選別完了', label='音声セグメントの話者名'),  # type: ignore
                gradio.Textbox(value='すべてのセグメントの選別を完了しました。Aivis のプロセスを終了してください。', label='音声セグメントの書き起こし文'),
            )

        # UI を更新
        return (
            gradio.Audio(
                value = segment_audio_paths[current_index],
                sources = [],
                type = 'filepath',
                label = segment_audio_paths[current_index].name,
                interactive = True,
                autoplay = True,
                waveform_options = WaveformOptions(sample_rate=44100),  # UI 上でトリミングした音声ファイルのサンプリングレート
            ),
            gradio.Dropdown(choices=choices, value=choices[0], label='音声セグメントの話者名'),  # type: ignore
            gradio.Textbox(value=segment_audio_transcripts[current_index], label='音声セグメントの書き起こし文'),
        )

    def OnReset(speaker_name: str) -> tuple[gradio.Audio, gradio.Textbox]:
        """ リセットボタンが押されたときの処理 """

        nonlocal current_index, segment_audio_paths, segment_audio_transcripts, choices

        # 話者名が空の場合は初期画面から「確定」を押して実行されたイベントなので、デフォルトのフォームを返す
        if speaker_name == '':
            return (
                gradio.Audio(
                    sources = [],
                    type = 'filepath',
                    interactive = True,
                    autoplay = True,
                    waveform_options = WaveformOptions(sample_rate=44100),  # UI 上でトリミングした音声ファイルのサンプリングレート
                ),
                gradio.Textbox(value='確定ボタンを押して、データセット作成を開始してください。', label='音声セグメントの書き起こし文'),
            )

        # 現在の current_index に応じて音声と書き起こし文をリセット
        return (
            gradio.Audio(
                value = segment_audio_paths[current_index],
                sources = [],
                type = 'filepath',
                label = segment_audio_paths[current_index].name,
                interactive = True,
                autoplay = True,
                waveform_options = WaveformOptions(sample_rate=44100),  # UI 上でトリミングした音声ファイルのサンプリングレート
            ),
            gradio.Textbox(value=segment_audio_transcripts[current_index], label='音声セグメントの書き起こし文'),
        )

    # Gradio UI の定義と起動
    with gradio.Blocks(css='.gradio-container { max-width: 768px !important; }') as gui:
        with gradio.Column():
            gradio.Markdown("""
                # Aivis - Create Datasets
                Tab キー / Shift + Tab キー を押すと、フォームやボタン間で素早くフォーカスを移動できます。
            """)
            audio_player = gradio.Audio(
                sources = [],
                type = 'filepath',
                interactive = True,
                autoplay = True,
                waveform_options = WaveformOptions(sample_rate=44100),  # UI 上でトリミングした音声ファイルのサンプリングレート
            )
            speaker_choice = gradio.Dropdown(choices=[], value='', label='音声セグメントの話者名')  # type: ignore
            transcript_box = gradio.Textbox(value='確定ボタンを押して、データセット作成を開始してください。', label='音声セグメントの書き起こし文')
            with gradio.Row():
                confirm_button = gradio.Button('確定', variant='primary')
                skip_button = gradio.Button('このデータを除外')
            confirm_button.click(
                fn = OnClick,
                inputs = [
                    audio_player,
                    speaker_choice,
                    transcript_box,
                ],
                outputs = [
                    audio_player,
                    speaker_choice,
                    transcript_box,
                ],
            )
            skip_button.click(
                # functools.partial() を使って OnClick() に is_skip=True を渡す
                fn = functools.partial(OnClick, is_skip=True),
                inputs = [
                    audio_player,
                    speaker_choice,
                    transcript_box,
                ],
                outputs = [
                    audio_player,
                    speaker_choice,
                    transcript_box,
                ],
            )
            reset_button = gradio.Button('音声と書き起こし文の変更をリセット')
            reset_button.click(
                fn = OnReset,
                inputs = [
                    speaker_choice,
                ],
                outputs = [
                    audio_player,
                    transcript_box,
                ],
            )

        # 0.0.0.0:7860 で Gradio UI を起動
        gui.launch(server_name='0.0.0.0', server_port=7860)


@app.command(help='Check dataset files and calculate total duration.')
def check_dataset(
    speaker_name: Annotated[str, typer.Argument(help='Speaker name.')],
):
    typer.echo('=' * utils.GetTerminalColumnSize())

    # バリデーション
    dataset_dir = constants.DATASETS_DIR / speaker_name
    if not dataset_dir.exists():
        typer.echo(f'Error: Speaker {speaker_name} not found.')
        typer.echo('=' * utils.GetTerminalColumnSize())
        sys.exit(1)

    # transcripts.list をパースして音声ファイル名と書き起こし文を取得
    ## 例: 0001.wav|SpeakerName|JP|こんにちは
    with open(dataset_dir / 'transcripts.list', mode='r', encoding='utf-8') as f:
        dataset_files_raw = f.read().splitlines()
        dataset_files = [i.split('|') for i in dataset_files_raw]

    typer.echo(f'Speaker: {speaker_name} / Directory: {dataset_dir}')
    typer.echo('=' * utils.GetTerminalColumnSize())

    # 各音声ファイルごとにループ
    total_audio_duration = 0.0
    for index, dataset_file in enumerate(dataset_files):
        if index > 0:
            typer.echo('-' * utils.GetTerminalColumnSize())
        dataset_file_path = constants.DATASETS_DIR / speaker_name / 'audios' / dataset_file[0]
        typer.echo(f'Dataset File : {dataset_file_path}')
        if not dataset_file_path.exists():
            typer.echo(f'Error: Dataset file {dataset_file_path} not found.')
        else:
            audio_duration = prepare.GetAudioFileDuration(dataset_file_path)
            total_audio_duration += audio_duration
            typer.echo(f'Duration     : {utils.SecondToTimeCode(audio_duration)}')
            typer.echo(f'Transcript   : {dataset_file[3]}')

    typer.echo('=' * utils.GetTerminalColumnSize())
    typer.echo(f'Total Files    : {len(dataset_files)}')
    typer.echo(f'Total Duration : {utils.SecondToTimeCode(total_audio_duration)}')
    typer.echo('=' * utils.GetTerminalColumnSize())


@app.command(help='Train model.')
def train(
    speaker_name: Annotated[str, typer.Argument(help='Speaker name.')],
    batch_size: Annotated[int, typer.Option(help='Training batch size.')] = 4,
    epochs: Annotated[Union[int, None], typer.Option(help='Training epochs. (Cannot be used with --steps)')] = None,
    steps: Annotated[Union[int, None], typer.Option(help='Training steps. (Cannot be used with --epochs)')] = None,
):
    typer.echo('=' * utils.GetTerminalColumnSize())

    # バリデーション
    dataset_dir = constants.DATASETS_DIR / speaker_name
    if not dataset_dir.exists():
        typer.echo(f'Error: Speaker {speaker_name} not found.')
        typer.echo('=' * utils.GetTerminalColumnSize())
        sys.exit(1)
    if epochs is not None and steps is not None:
        typer.echo(f'Error: --epochs and --steps cannot be used together.')
        typer.echo('=' * utils.GetTerminalColumnSize())
        sys.exit(1)
    if epochs is None and steps is None:
        typer.echo(f'Error: --epochs or --steps must be specified.')
        typer.echo('=' * utils.GetTerminalColumnSize())
        sys.exit(1)

    # transcripts.list をパースしてデータセットの音声ファイルの総数を取得
    with open(dataset_dir / 'transcripts.list', mode='r', encoding='utf-8') as f:
        dataset_files_raw = f.read().splitlines()
        dataset_files = [i.split('|') for i in dataset_files_raw]
        dataset_files_count = len(dataset_files)

    # もし --epochs が指定されている場合、バッチサイズ・データセットの総数から自動的にステップ数を計算
    if epochs is not None:
        steps = math.ceil((dataset_files_count / batch_size) * epochs)

    # もし --steps が指定されている場合、バッチサイズ・データセットの総数から自動的にエポック数を計算
    if steps is not None:
        epochs = math.ceil(steps / (dataset_files_count / batch_size)) + 1  # モデルを確実に保存するため +1 しておく

    typer.echo(f'Speaker: {speaker_name} / Directory: {dataset_dir} (Total {dataset_files_count} files)')
    typer.echo(f'Batch Size: {batch_size} / Epochs: {epochs} / Steps: {steps}')
    typer.echo('=' * utils.GetTerminalColumnSize())

    # Bert-VITS2 のデータセットディレクトリを作成
    bert_vits2_dataset_dir = constants.BERT_VITS2_DIR / 'Data'
    bert_vits2_dataset_dir.mkdir(parents=True, exist_ok=True)

    # 事前学習済みモデルがまだダウンロードされていなければダウンロード
    ## ダウンロード中に実行を中断するとダウンロード途中のロードできない事前学習済みモデルが残ってしまう
    ## 基本ダウンロード中に実行を中断すべきではないが、万が一そうなった場合は手動でダウンロード途中のモデルを削除してから再実行する必要がある
    download_base_url = 'https://huggingface.co/Stardust-minus/Bert-VITS2-Japanese-Extra/resolve/main/'
    if not (constants.CACHE_DIR / 'D_0.pth').exists():
        typer.echo('Downloading pretrained model (D_0.pth) ...')
        utils.DownloadFile(download_base_url + 'D_0.pth', constants.CACHE_DIR / 'D_0.pth')
    if not (constants.CACHE_DIR / 'G_0.pth').exists():
        typer.echo('Downloading pretrained model (G_0.pth) ...')
        utils.DownloadFile(download_base_url + 'G_0.pth', constants.CACHE_DIR / 'G_0.pth')
    if not (constants.CACHE_DIR / 'WD_0.pth').exists():
        typer.echo('Downloading pretrained model (WD_0.pth) ...')
        utils.DownloadFile(download_base_url + 'WD_0.pth', constants.CACHE_DIR / 'WD_0.pth')

    # 既に Bert-VITS2/Data/(話者名)/audios/ が存在する場合は一旦削除
    ## 同一のデータセットでもう一度学習を回す際、Bert 関連の中間ファイルを削除して再生成されるようにする
    if (bert_vits2_dataset_dir / speaker_name / 'audios').exists():
        shutil.rmtree(bert_vits2_dataset_dir / speaker_name / 'audios')
    ## 再度空のディレクトリを作成
    (bert_vits2_dataset_dir / speaker_name / 'audios').mkdir(parents=True, exist_ok=True)

    # 既に Bert-VITS2/Data/(話者名)/filelists/ が存在する場合は一旦削除
    ## 同一のデータセットでもう一度学習を回す際、書き起こしデータの中間ファイルを削除して再生成されるようにする
    if (bert_vits2_dataset_dir / speaker_name / 'filelists').exists():
        shutil.rmtree(bert_vits2_dataset_dir / speaker_name / 'filelists')
    ## 再度空のディレクトリを作成
    (bert_vits2_dataset_dir / speaker_name / 'filelists').mkdir(parents=True, exist_ok=True)

    # 指定されたデータセットを Bert-VITS2 のデータセットディレクトリにコピー
    ## ex: 04-Datasets/(話者名)/audios/ -> Bert-VITS2/Data/(話者名)/audios/wavs/
    ## ex: 04-Datasets/(話者名)/transcripts.list -> Bert-VITS2/Data/(話者名)/filelists/transcripts.list
    typer.echo('Copying dataset files...')
    shutil.copytree(dataset_dir / 'audios', bert_vits2_dataset_dir / speaker_name / 'audios' / 'wavs')
    shutil.copyfile(dataset_dir / 'transcripts.list', bert_vits2_dataset_dir / speaker_name / 'filelists' / 'transcripts.list')

    # 書き起こし文ファイル内の音声ファイル名を Data/(話者名)/audios/wavs/ からのパスに変更
    ## 例: 0001.wav|SpeakerName|JP|こんにちは → Data/SpeakerName/audios/wavs/0001.wav|SpeakerName|JP|こんにちは
    with open(bert_vits2_dataset_dir / speaker_name / 'filelists' / 'transcripts.list', 'r', encoding='utf-8') as f:
        transcripts_list = f.read()
    with open(bert_vits2_dataset_dir / speaker_name / 'filelists' / 'transcripts.list', 'w', encoding='utf-8') as f:
        f.write(re.sub(r'(.*\.wav)', f'Data/{speaker_name}/audios/wavs/\\1', transcripts_list))

    # ダウンロードした事前学習済みモデルを Bert-VITS2/Data/(話者名)/models/ にコピー
    ## モデル学習の際にこれらのファイルは上書きされてしまうため、シンボリックリンクではなくコピーする
    if not (bert_vits2_dataset_dir / speaker_name / 'models').exists():
        typer.echo('Copying pretrained model files...')
        (bert_vits2_dataset_dir / speaker_name / 'models').mkdir(parents=True, exist_ok=True)
        ## ex: Bert-VITS2/Data/G_0.pth -> Bert-VITS2/Data/(話者名)/models/G_0.pth
        if not (bert_vits2_dataset_dir / speaker_name / 'models' / 'D_0.pth').exists():
            shutil.copyfile(constants.CACHE_DIR / 'D_0.pth', bert_vits2_dataset_dir / speaker_name / 'models' / 'D_0.pth')
        if not (bert_vits2_dataset_dir / speaker_name / 'models' / 'G_0.pth').exists():
            shutil.copyfile(constants.CACHE_DIR / 'G_0.pth', bert_vits2_dataset_dir / speaker_name / 'models' / 'G_0.pth')
        if not (bert_vits2_dataset_dir / speaker_name / 'models' / 'WD_0.pth').exists():
            shutil.copyfile(constants.CACHE_DIR / 'WD_0.pth', bert_vits2_dataset_dir / speaker_name / 'models' / 'WD_0.pth')

    # Bert-VITS2/configs/config.json を Bert-VITS2/Data/(話者名)/config.json にコピー
    ## モデル学習の際にこれらのファイルは上書きされてしまうため、シンボリックリンクではなくコピーする
    if not (bert_vits2_dataset_dir / speaker_name / 'config.json').exists():
        typer.echo('Copying model config file...')
        shutil.copyfile(constants.BERT_VITS2_DIR / 'configs' / 'config.json', bert_vits2_dataset_dir / speaker_name / 'config.json')

    # コピーした config.json の epochs と batch_size とを指定された値に変更
    with open(bert_vits2_dataset_dir / speaker_name / 'config.json', mode='r', encoding='utf-8') as f:
        config = json.load(f)
    config['train']['epochs'] = epochs
    config['train']['batch_size'] = batch_size
    with open(bert_vits2_dataset_dir / speaker_name / 'config.json', mode='w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Bert-VITS2/default_config.yml を Bert-VITS2/config.yml にコピー
    ## 学習対象のデータセット名を変更する必要があるため、既に config.yml が存在する場合も上書きする
    typer.echo('Copying default_config.yml to config.yml...')
    shutil.copyfile(constants.BERT_VITS2_DIR / 'default_config.yml', constants.BERT_VITS2_DIR / 'config.yml')

    # config.yml 内の dataset_path: "Data/MySpeaker" を dataset_path: "Data/(話者名)" に変更
    ## 正規表現で置換する
    with open(constants.BERT_VITS2_DIR / 'config.yml', mode='r', encoding='utf-8') as f:
        config_yml = f.read()
    config_yml = re.sub(r'dataset_path: "Data/.*"', f'dataset_path: "Data/{speaker_name}"', config_yml)
    with open(constants.BERT_VITS2_DIR / 'config.yml', mode='w', encoding='utf-8') as f:
        f.write(config_yml)
    typer.echo('=' * utils.GetTerminalColumnSize())

    # Bert-VITS2/preprocess_text.py を実行
    typer.echo('Running preprocess_text.py...')
    typer.echo('-' * utils.GetTerminalColumnSize())
    subprocess.run(
        ['python', constants.BERT_VITS2_DIR / 'preprocess_text.py'],
        cwd = constants.BERT_VITS2_DIR,  # カレントディレクトリを Bert-VITS2/ に変更しないと実行できない
        check = True,
    )
    typer.echo('=' * utils.GetTerminalColumnSize())

    # Bert-VITS2/bert_gen.py を実行
    typer.echo('Running bert_gen.py...')
    typer.echo('-' * utils.GetTerminalColumnSize())
    subprocess.run(
        ['python', constants.BERT_VITS2_DIR / 'bert_gen.py'],
        cwd = constants.BERT_VITS2_DIR,  # カレントディレクトリを Bert-VITS2/ に変更しないと実行できない
        check = True,
    )
    typer.echo('=' * utils.GetTerminalColumnSize())

    # Bert-VITS2/clap_gen.py を実行
    typer.echo('Running clap_gen.py...')
    typer.echo('-' * utils.GetTerminalColumnSize())
    subprocess.run(
        ['python', constants.BERT_VITS2_DIR / 'clap_gen.py'],
        cwd = constants.BERT_VITS2_DIR,  # カレントディレクトリを Bert-VITS2/ に変更しないと実行できない
        check = True,
    )
    typer.echo('=' * utils.GetTerminalColumnSize())

    # 学習を開始 (Bert-VITS2/train_ms.py を実行)
    typer.echo('Training started.')
    typer.echo('-' * utils.GetTerminalColumnSize())
    try:
        subprocess.run(
            ['python', constants.BERT_VITS2_DIR / 'train_ms.py'],
            cwd = constants.BERT_VITS2_DIR,  # カレントディレクトリを Bert-VITS2/ に変更しないと実行できない
            check = True,
        )
    except subprocess.CalledProcessError as ex:
        typer.echo('-' * utils.GetTerminalColumnSize())
        typer.echo(f'Training failed. (Process exited with code {ex.returncode})')
        typer.echo('=' * utils.GetTerminalColumnSize())
        sys.exit(1)
    typer.echo('-' * utils.GetTerminalColumnSize())
    typer.echo('Training finished.')
    typer.echo('=' * utils.GetTerminalColumnSize())


@app.command(help='Infer model.')
def infer(
    speaker_name: Annotated[str, typer.Argument(help='Speaker name.')],
    model_step: Annotated[Optional[int], typer.Option(help='Model step. (Default: Largest step)')] = None,
):
    typer.echo('=' * utils.GetTerminalColumnSize())

    # バリデーション
    model_dir = constants.BERT_VITS2_DIR / 'Data' / speaker_name
    if not model_dir.exists():
        typer.echo(f'Error: Speaker {speaker_name} not found.')
        typer.echo('=' * utils.GetTerminalColumnSize())
        sys.exit(1)

    # モデルファイルを探す
    # 指定されていなければ最大のステップのモデルを探す
    ## モデルは 1000 ステップごとに保存されており、G_(ステップ数).pth のファイル名フォーマットで保存されている
    ## 例: G_0.pth / G_1000.pth / G_2000.pth / G_3000.pth
    if model_step is None:
        model_step = 0
        for model_file in (model_dir / 'models').glob('G_*.pth'):
            step = int(re.sub(r'\D', '', model_file.stem))
            if step > model_step:
                model_step = step
        if (model_dir / 'models' / f'G_{model_step}.pth').exists():
            model_file = model_dir / 'models' / f'G_{model_step}.pth'
        else:
            typer.echo(f'Error: Model file {model_dir / "models" / f"G_{model_step}.pth"} not found.')
            typer.echo('=' * utils.GetTerminalColumnSize())
            sys.exit(1)

    # ステップ数が指定されている場合はそのステップのモデルを探す
    else:
        model_file = model_dir / 'models' / f'G_{model_step}.pth'
        if not model_file.exists():
            typer.echo(f'Error: Model file {model_file} not found.')
            typer.echo('=' * utils.GetTerminalColumnSize())
            sys.exit(1)

    typer.echo(f'Speaker: {speaker_name} / Model Directory: {model_dir}')
    typer.echo(f'Model File: {model_file}')
    typer.echo('=' * utils.GetTerminalColumnSize())

    # config.yml を正規表現で書き換える
    ## dataset_path: ".*" を dataset_path: "Data/(話者名)" に書き換える
    ## model: "models/.*" を model: "models/G_(ステップ数).pth" に書き換える
    with open(constants.BERT_VITS2_DIR / 'config.yml', mode='r', encoding='utf-8') as f:
        config_yml = f.read()
    config_yml = re.sub(r'dataset_path: ".*"', f'dataset_path: "Data/{speaker_name}"', config_yml)
    config_yml = re.sub(r'model: "models/.*"', f'model: "models/G_{model_step}.pth"', config_yml)
    with open(constants.BERT_VITS2_DIR / 'config.yml', mode='w', encoding='utf-8') as f:
        f.write(config_yml)

    # Bert-VITS2/webui.py を実行
    typer.echo('Running Infer Web UI...')
    typer.echo('-' * utils.GetTerminalColumnSize())
    subprocess.run(
        ['python', constants.BERT_VITS2_DIR / 'webui.py'],
        cwd = constants.BERT_VITS2_DIR,  # カレントディレクトリを Bert-VITS2/ に変更しないと実行できない
        check = True,
    )
    typer.echo('=' * utils.GetTerminalColumnSize())


@app.command(help='Show version.')
def version():
    typer.echo(f'Aivis version {__version__}')


if __name__ == '__main__':
    app()
