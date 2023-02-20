
import re
from ffmpeg_normalize import FFmpegNormalize
from inaSpeechSegmenter import Segmenter
from pathlib import Path
from pydub import AudioSegment
from typing import cast, Literal

from Aivis import utils


def GetAudioFileDuration(file_path: Path) -> float:
    """
    音声ファイルの長さを取得する

    Args:
        file_path (Path): 音声ファイルのパス

    Returns:
        float: 音声ファイルの長さ (秒)
    """

    # 音声ファイルを読み込む
    audio = AudioSegment.from_file(file_path)

    # 音声ファイルの長さを取得する
    return audio.duration_seconds


def SliceAudioFile(src_file_path: Path, dst_file_path: Path, start: float, end_min: float, end_max: float) -> None:
    """
    音声ファイルの一部を切り出して出力する

    Args:
        src_file_path (Path): 切り出し元の音声ファイルのパス
        dst_file_path (Path): 切り出し先の音声ファイルのパス
        start (float): 切り出し開始時間 (秒)
        end_min (float): 切り出し終了時間 (最小) (秒)
        end_max (float): 切り出し終了時間 (最大) (秒)
    """

    # 開始時刻ちょうどから切り出すと子音が切れてしまうことがあるため、開始時刻の 0.1 秒前から切り出す
    start = max(0, start - 0.1)

    # 音声ファイルを読み込む
    audio = AudioSegment.from_file(src_file_path)

    # 一旦終了時間に余裕を持たせて音声ファイルを切り出す
    ## 一旦次の文の開始直前までを切り出す
    ## このあと、無音区間を検出して切り出し終了時間を調整する
    dst_file_path = dst_file_path.with_suffix('.wav')
    sliced_audio = audio[start * 1000:end_max * 1000]
    sliced_audio.export(dst_file_path, format='wav')

    # 一旦切り出して出力した後のファイルでの end_min と end_max
    ## この時点では終了時間 sliced_end_max 秒まで切り出されているので、ここから無音区間次第で最大で終了時間 sliced_end_min 秒まで切り詰める
    sliced_end_min = end_min - start
    sliced_end_max = end_max - start

    # inaSpeechSegmenter で無音区間を検出する
    ## 'sm' は入力信号を音声区間 (speeech) / 音楽区間 (music) / 無音区間 (noEnergy) にラベル付けしてくれる
    ## ref: https://qiita.com/shimajiroxyz/items/de213cd333e7bf846781
    segmenter = Segmenter(vad_engine='sm', detect_gender=False)
    segments = cast(list[tuple[Literal['speech', 'music', 'noEnergy'], float, float]], segmenter(str(dst_file_path)))

    # まず無音区間の開始時間、終了時間だけを抽出する
    no_energy_segments = [(segment[1], segment[2]) for segment in segments if segment[0] == 'noEnergy']

    # 次に、sliced_end_min 以降でかつ一番近い無音区間の開始時間を探す
    ## なければ sliced_end_max を採用する
    sliced_end = sliced_end_max
    for no_energy_segment in no_energy_segments:
        # 無音区間の開始時間が sliced_end_min 以降でかつ sliced_end 以前の場合のみ採用する
        if no_energy_segment[0] >= sliced_end_min and no_energy_segment[0] < sliced_end:
            sliced_end = no_energy_segment[0]

    print(f'End Time (Min): {sliced_end_min:.3f} / End Time (Max): {sliced_end_max:.3f} / Confirmed End Time: {sliced_end:.3f}')
    print(f'Segment Range: {utils.SecondToTimeCode(start)} - {utils.SecondToTimeCode(start + sliced_end)}')

    # 改めて音声ファイルを切り出す
    ## 開始位置の調整は不要なので、切り出し終了時間のみを指定する
    ## 既存のファイルは上書きされる
    sliced_audio = AudioSegment.from_file(dst_file_path)
    new_sliced_audio = sliced_audio[0:sliced_end * 1000]

    # 音声ファイルを出力する
    ## 一旦一時ファイルに出力したあと、FFmpeg で 44.1kHz 16bit モノラルの wav 形式に変換する
    ## 基本この時点で 44.1kHz 16bit にはなっているはずだが、音声チャンネルはステレオのままなので、ここでモノラルにダウンミックスする
    ## さらに音声ファイルの音量をノーマライズし、よりデータセットにふさわしい形にする
    ## これでデータセット向けの一文ごとの音声ファイルが完成する
    dst_file_path_temp = dst_file_path.with_suffix('.temp.wav')
    new_sliced_audio.export(dst_file_path_temp, format='wav')
    normalize = FFmpegNormalize(
        target_level = -14,  # -14LUFS にノーマライズする (YouTube が -14LUFS なのを参考にした)
        sample_rate = 44100,
        audio_codec = 'pcm_s16le',
        extra_output_options = ['-ac', '1'],
        output_format = 'wav',
    )
    normalize.add_media_file(str(dst_file_path_temp), str(dst_file_path))
    normalize.run_normalization()

    # 一時ファイルを削除
    dst_file_path_temp.unlink()


def PrepareText(text: str) -> str:
    """
    Whisper で書き起こされたテキストをより適切な形に前処理する
    (Whisper の書き起こし結果にはガチャがあり、句読点が付く場合と付かない場合があるため、前処理が必要)

    Args:
        text (str): Whisper で書き起こされたテキスト

    Returns:
        str: 前処理されたテキスト
    """

    # 前後の空白を削除する
    text = text.strip()

    # 末尾に記号がついていない場合は 。を追加する
    if text[-1] not in ['、', '。', '!', '?', '！', '？']:
        text = text + '。'

    # 同じ文字が4文字以上続いていたら (例: ～～～～～～～～！！)、2文字にする (例: ～～！！)
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)

    # 中間にある空白文字 (半角/全角の両方) を 、に置換する
    text = re.sub(r'[ 　]', '、', text)

    return text
