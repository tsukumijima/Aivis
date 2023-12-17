#!/usr/bin/env python3

# FutureWarning / RuntimeWarning / UserWarning を抑制する
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import json
import re
import shutil
import sys
import typer
from pathlib import Path
from typing import Annotated, Any, cast

from Aivis import __version__
from Aivis import constants
from Aivis import demucs
from Aivis import prepare
from Aivis import utils


app = typer.Typer(help='Aivis: AI Voice Imitation System')

@app.command(help='Create audio segments from audio sources.')
def create_segments(
    model_name: Annotated[constants.ModelNameType, typer.Option(help='Model name.')] = constants.ModelNameType.large_v3,
    force_transcribe: Annotated[bool, typer.Option(help='Force Whisper to transcribe audio files.')] = False,
):
    # このサブコマンドでしか利用せず、かつ比較的インポートが重いモジュールはここでインポートする
    import faster_whisper
    import stable_whisper

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

    model: faster_whisper.WhisperModel | None = None

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
                typer.echo(f'Whisper model loading... (Model: {model_name.value})')
                model = stable_whisper.load_faster_whisper(
                    model_name.value,
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
            real_output_audio_file = prepare.SliceAudioFile(voices_file, output_audio_file, segment_start, segment_end)

            typer.echo(f'File {real_output_audio_file} saved.')
            count += 1

    typer.echo('=' * utils.GetTerminalColumnSize())
    typer.echo('All files segmentation done.')
    typer.echo('=' * utils.GetTerminalColumnSize())


@app.command(help='Create datasets from audio segments.')
def create_datasets(
    segments_dir_name: Annotated[str, typer.Argument(help='Segments directory name.')],
    speaker_names: Annotated[str, typer.Argument(help='Speaker name. (Comma separated)')],
):
    # このサブコマンドでしか利用せず、かつ比較的インポートが重いモジュールはここでインポートする
    import gradio

    typer.echo('=' * utils.GetTerminalColumnSize())

    # バリデーション
    if (constants.SEGMENTS_DIR / segments_dir_name).exists() is False:
        typer.echo(f'Error: {segments_dir_name} is not directory.')
        typer.echo('=' * utils.GetTerminalColumnSize())
        return
    if speaker_names == '':
        typer.echo(f'Error: Speaker names is empty.')
        typer.echo('=' * utils.GetTerminalColumnSize())
        return

    # 出力後のデータセットの出力先ディレクトリがなければ作成
    speaker_name_list = speaker_names.split(',')
    for speaker in speaker_name_list:
        output_dir = constants.DATASETS_DIR / speaker
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            typer.echo(f'Speaker: {speaker} / Folder: {output_dir} created.')
        else:
            typer.echo(f'Speaker: {speaker} / Folder: {output_dir} already created.')
    typer.echo('=' * utils.GetTerminalColumnSize())

    # 03-Segments/(指定されたディレクトリ名、通常は音声ファイル名と同じ) フォルダ以下のセグメント化された音声ファイルを取得
    ## 拡張子は .wav
    ## glob() の結果は順序がバラバラなのでアルファベット順にソートする
    segment_audio_paths = sorted(list((constants.SEGMENTS_DIR / segments_dir_name).glob('**/*.wav')))

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
            with open(segment_audio_transcript_txt, 'r', encoding='utf-8') as f:
                segment_audio_transcript = f.read()

        # 書き起こし文をリストに追加
        segment_audio_transcripts.append(segment_audio_transcript)

    # 現在処理中の音声ファイルのインデックスと音声ファイルのパスと書き起こし文
    current_index = 0

    # セレクトボックスの選択肢
    choices = ['このセグメントをデータセットから除外する'] + speaker_name_list

    # 出力ファイルの連番
    output_audio_count: dict[str, int] = {}
    for speaker in speaker_name_list:
        # 既にそのディレクトリに存在するファイルの中で連番が一番大きいものを取得し、それに 1 を足したものを初期値とする
        output_audio_count[speaker] = max([
            int(re.sub(r'\D', '', i.stem)) for i in (constants.DATASETS_DIR / speaker / 'audios' / 'wavs').glob('*.wav')
        ], default=0) + 1

    def OnClick(
        segment_audio_path_str: str,
        speaker_name: str,
        transcript: str,
    ) -> tuple[gradio.Audio, gradio.Textbox, gradio.Dropdown]:
        """ 確定ボタンが押されたときの処理 """

        nonlocal current_index, segment_audio_paths, segment_audio_transcripts, choices, output_audio_count
        segment_audio_path = Path(segment_audio_path_str)
        typer.echo('-' * utils.GetTerminalColumnSize())
        typer.echo(f'Segment File : {segment_audio_path.name}')
        typer.echo(f'Speaker Name : {speaker_name}')
        typer.echo(f'Transcript   : {transcript}')

        # "このセグメントをデータセットから除外する" が選択された場合はスキップ
        if speaker_name != 'このセグメントをデータセットから除外する':

            # データセットに編集後の音声ファイルを保存 (書き起こし文はファイル名が長くなるので含まず、別途ファイルに保存する)
            ## Gradio の謎機能で、GUI でトリムした編集後の一次ファイルが segment_audio_path_str として渡されてくる
            audio_output_dir = constants.DATASETS_DIR / speaker_name / 'audios' / 'wavs'
            audio_output_dir.mkdir(parents=True, exist_ok=True)
            output_path = audio_output_dir / f'{output_audio_count[speaker_name]:04}.wav'
            output_audio_count[speaker_name] += 1  # 連番をインクリメント
            shutil.copyfile(segment_audio_path, output_path)
            typer.echo(f'File {output_path} saved.')

            # 音声ファイルのパスと書き起こし文のパスのペアを speaker.list に順次追記
            text_list_path = constants.DATASETS_DIR / speaker_name / 'filelists' / 'speaker.list'
            if not text_list_path.exists():
                text_list_path.parent.mkdir(parents=True, exist_ok=True)
                text_list_path.touch()
            with open(text_list_path, 'a', encoding='utf-8') as f:
                f.write(f'Data/{speaker_name}/audios/wavs/{output_path.name}|{speaker_name}|JP|{transcript}\n')
            typer.echo(f'File {text_list_path} updated.')
            typer.echo('-' * utils.GetTerminalColumnSize())

        else:
            typer.echo('Segment file skipped.')
            typer.echo('-' * utils.GetTerminalColumnSize())

        # 次の処理対象のファイルのインデックス
        current_index += 1

        # 次の処理対象のファイルがない場合は終了
        if current_index < 0 or current_index >= len(segment_audio_paths):
            typer.echo('=' * utils.GetTerminalColumnSize())
            typer.echo('All files processed.')
            typer.echo('=' * utils.GetTerminalColumnSize())
            sys.exit(0)

        # UI を更新
        return (
            gradio.Audio(
                value = segment_audio_paths[current_index],
                sources = [],
                type = 'filepath',
                label = segment_audio_paths[current_index].name,
                interactive = True,
                autoplay = True,
            ),
            gradio.Dropdown(choices=choices, value=choices[0], label='音声セグメントの話者名'),  # type: ignore
            gradio.Textbox(value=segment_audio_transcripts[current_index], label='音声セグメントの書き起こし文'),
        )

    # Gradio のバグで UI 上でトリミングした音声セグメントのサンプルレートが 8000Hz になってしまう問題のワークアラウンド
    ## 本来は WaveSurfer.js の初期化時にサンプルレートを指定すべきところ指定されておらずデフォルト値でデコードされてしまい、
    ## トリミングする際も 8000Hz のデータが使われてしまうことが原因なので、強引にフロントエンドのファイルを書き換える
    ## sampleRate: 8000, -> sampleRate: 44100, (音声セグメントのサンプルレートは 44100Hz 固定なのでこれで動く)
    ## ref: https://github.com/gradio-app/gradio/issues/6567#issuecomment-1853392537
    index_js_path = Path(gradio.__file__).parent / 'templates/frontend/assets/index-84ec2915.js'
    with open(index_js_path, 'r', encoding='utf-8') as f:
        index_js = f.read()
    index_js = index_js.replace('sampleRate: 8000,', 'sampleRate: 44100,')
    with open(index_js_path, 'w', encoding='utf-8') as f:
        f.write(index_js)

    # Gradio UI の定義と起動
    with gradio.Blocks() as gui:
        with gradio.Column():
            gradio.Markdown("""
                # Aivis - Create Datasets
            """)
            audio_player = gradio.Audio(
                value = segment_audio_paths[current_index],
                sources = [],
                type = 'filepath',
                label = segment_audio_paths[current_index].name,
                interactive = True,
                autoplay = True,
            )
            speaker_choice = gradio.Dropdown(choices=choices, value=choices[0], label='音声セグメントの話者名')  # type: ignore
            transcript_box = gradio.Textbox(value=segment_audio_transcripts[current_index], label='音声セグメントの書き起こし文')
            confirm_button = gradio.Button('確定')

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

        # 0.0.0.0:7860 で Gradio UI を起動
        gui.launch(server_name='0.0.0.0', server_port=7860)


@app.command(help='Show version.')
def version():
    typer.echo(f'Aivis version {__version__}')


if __name__ == '__main__':
    app()
