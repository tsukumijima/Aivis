
import errno
import librosa
import pyloudnorm
import re
import regex
import shutil
import soundfile
import subprocess
import sys
import tempfile
import typer
from pathlib import Path
from pydub import AudioSegment


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


def SliceAudioFile(src_file_path: Path, dst_file_path: Path, start: float, end: float, trim_silence: bool) -> Path:
    """
    音声ファイルの一部を切り出して出力する
    trim_silence=True のときは追加で切り出した音声ファイルの前後の無音区間が削除される

    Args:
        src_file_path (Path): 切り出し元の音声ファイルのパス
        dst_file_path (Path): 切り出し先の音声ファイルのパス
        start (float): 切り出し開始時間 (秒)
        end (float): 切り出し終了時間 (秒)
        trim_silence (bool): 前後の無音区間を削除するかどうか
    """

    # 一時保存先のテンポラリファイル
    ## Windows だと /tmp/ が使えないので NamedTemporaryFile を使う
    ## src_file_path --切り出し--> temp1 --モノラル化--> temp2 --ノーマライズ--> temp3 --無音区間削除--> temp4 --リネーム--> dst_file_path
    dst_file_path_temp1 = Path(tempfile.NamedTemporaryFile(suffix='.wav').name)
    dst_file_path_temp2 = Path(tempfile.NamedTemporaryFile(suffix='.wav').name)
    dst_file_path_temp3 = Path(tempfile.NamedTemporaryFile(suffix='.wav').name)
    dst_file_path_temp4 = Path(tempfile.NamedTemporaryFile(suffix='.wav').name)

    # 開始時刻ちょうどから切り出すと子音が切れてしまうことがあるため、開始時刻の 0.1 秒前から切り出す
    start = max(0, start - 0.1)

    # 音声ファイルを読み込む
    audio = AudioSegment.from_file(src_file_path)

    # 音声ファイルを切り出す
    sliced_audio = audio[start * 1000:end * 1000]
    sliced_audio.export(dst_file_path_temp1, format='wav')

    # FFmpeg で 44.1kHz 16bit モノラルの wav 形式に変換する
    ## 基本この時点で 44.1kHz 16bit にはなっているはずだが、音声チャンネルはステレオのままなので、ここでモノラルにダウンミックスする
    subprocess.run([
        'ffmpeg',
        '-y',
        '-i', str(dst_file_path_temp1),
        '-ac', '1',
        '-ar', '44100',
        '-acodec', 'pcm_s16le',
        str(dst_file_path_temp2),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # pyloudnorm で音声ファイルをノーマライズ（ラウドネス正規化）する
    ## FFmpeg でのモノラルへのダウンミックスで音量が変わる可能性も無くはないため、念のためダウンミックス後にノーマライズを行うように実装している
    LoudnessNorm(dst_file_path_temp2, dst_file_path_temp3, loudness=-23.0)  # -23LUFS にノーマライズする

    if trim_silence is True:
        # 最後に前後の無音区間を librosa を使って削除する
        ## sr=None を指定して、音声ファイルのサンプリングレートをそのまま維持して読み込む (指定しないと 22050Hz になる…)
        ## 無音区間は dB 基準なのでノーマライズ後に実行した方が望ましい
        y, sr = librosa.load(dst_file_path_temp3, sr=None)  # type: ignore
        y, _ = librosa.effects.trim(y, top_db=30)
        soundfile.write(dst_file_path_temp4, y, sr)
    else:
        # 無音区間は削除せずにそのままコピーする
        shutil.copyfile(dst_file_path_temp3, dst_file_path_temp4)

    # 最後にファイルを dst_file_path にコピーする
    try:
        shutil.copyfile(dst_file_path_temp4, dst_file_path)
    except OSError as ex:
        # 万が一ファイル名が最大文字数を超える場合は、ファイル名を短くする
        ## 87文字は、Linux のファイル名の最大バイト数 (255B) から、拡張子 (.wav) を引いた 251B に入る UTF-8 の最大文字数
        ## NTFS のファイル名の最大文字数は 255 文字なので (バイト単位ではない) 、Windows でも問題ないはず
        if ex.errno == errno.ENAMETOOLONG:
            # ファイル名を短くした上でコピーする
            dst_file_path_new = dst_file_path.with_name(dst_file_path.stem[:87] + dst_file_path.suffix)
            shutil.copyfile(dst_file_path_temp4, dst_file_path_new)
            typer.echo('Warning: File name is too long. Truncated.')
            # フルの書き起こし文にアクセスできるように、別途テキストファイルに書き起こし文を保存する
            with open(dst_file_path_new.with_suffix('.txt'), mode='w', encoding='utf-8') as f:
                transcript = re.sub(r'^\d+_', '', dst_file_path.stem)
                f.write(transcript)
            # ファイル名からの書き起こし文の取得が終わったので、dst_file_path を上書きする
            dst_file_path = dst_file_path_new
        # Windows でファイル名に使用できない文字が含まれている場合は、ファイル名から使用できない文字を置換する
        ## Windows はファイル名に使用できない文字が多い
        elif ex.errno == errno.EINVAL and sys.platform == 'win32':
            # ファイル名に使用できない文字を置換する
            dst_file_path_new = dst_file_path.with_name(re.sub(r'[\\/:*?"<>|]', '_', dst_file_path.stem) + dst_file_path.suffix)
            shutil.copyfile(dst_file_path_temp4, dst_file_path_new)
            typer.echo('Warning: File name contains invalid characters. Replaced.')
            # フルの書き起こし文にアクセスできるように、別途テキストファイルに書き起こし文を保存する
            with open(dst_file_path_new.with_suffix('.txt'), mode='w', encoding='utf-8') as f:
                transcript = re.sub(r'^\d+_', '', dst_file_path.stem)
                f.write(transcript)
            # ファイル名からの書き起こし文の取得が終わったので、dst_file_path を上書きする
            dst_file_path = dst_file_path_new
        else:
            raise ex

    # 一時ファイルを削除
    dst_file_path_temp1.unlink()
    dst_file_path_temp2.unlink()
    dst_file_path_temp3.unlink()
    dst_file_path_temp4.unlink()

    return dst_file_path


def LoudnessNorm(input: Path, output: Path, peak: float = -1.0, loudness: float = -23.0, block_size : float = 0.400) -> None:
    """
    音声ファイルに対して、ラウドネス正規化（ITU-R BS.1770-4）を実行する
    ref: https://github.com/fishaudio/audio-preprocess/blob/main/fish_audio_preprocess/utils/loudness_norm.py#L9-L33

    Args:
        input: 入力音声ファイル
        output: 出力音声ファイル
        peak: 音声を N dB にピーク正規化する. Defaults to -1.0.
        loudness: 音声を N dB LUFS にラウドネス正規化する. Defaults to -23.0.
        block_size: ラウドネス測定用のブロックサイズ. Defaults to 0.400. (400 ms)

    Returns:
        ラウドネス正規化された音声データ
    """

    # 音声ファイルを読み込む
    audio, rate = soundfile.read(str(input))

    # ノーマライズを実行
    audio = pyloudnorm.normalize.peak(audio, peak)
    meter = pyloudnorm.Meter(rate, block_size=block_size)  # create BS.1770 meter
    try:
        _loudness = meter.integrated_loudness(audio)
        audio = pyloudnorm.normalize.loudness(audio, _loudness, loudness)
    except ValueError:
        pass

    # 音声ファイルを出力する
    soundfile.write(str(output), audio, rate)


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

    # 入力テキストに 1 つでもひらがな・カタカナ・漢字が含まれる場合のみ、日本語として処理する
    # ref: https://note.nkmk.me/python-re-regex-character-type/
    is_japanese = False
    hiragana_katakana_kanji_pattern = regex.compile(r'\p{Hiragana}|\p{Katakana}|\p{Han}')
    if hiragana_katakana_kanji_pattern.search(text):
        is_japanese = True

    # 半角の ､｡!? を 全角の 、。！？ に置換する
    if is_japanese is True:
        text = text.replace('､', '、')
        text = text.replace('｡', '。')
        text = text.replace('!', '！')
        text = text.replace('?', '？')

    # 全角の 、。！？ の後に半角スペースがある場合は削除する
    if is_japanese is True:
        text = text.replace('、 ', '、')
        text = text.replace('。 ', '。')
        text = text.replace('！ ', '！')
        text = text.replace('？ ', '？')

    # 末尾に記号がついていない場合は 。を追加する
    if is_japanese is True:
        if text[-1] not in ['、', '。','！', '？']:
            text = text + '。'
    else:
        if text[-1] not in ['.', '!', '?']:
            text = text + '.'

    # 先頭に 、。！？ がある場合は削除する
    if is_japanese is True:
        text = re.sub(r'^[、。！？]+', '', text)
    else:
        text = re.sub(r'^[,.!?]+', '', text)

    # 同じ文字が4文字以上続いていたら (例: ～～～～～～～～！！)、2文字にする (例: ～～！！)
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)

    # 中間にある空白文字 (半角/全角の両方) を 、に置換する
    if is_japanese is True:
        text = re.sub(r'[ 　]', '、', text)

    # （）や【】「」で囲われた文字列を削除する
    text = re.sub(r'（.*?）', '', text)
    text = re.sub(r'【.*?】', '', text)
    text = re.sub(r'「.*?」', '', text)

    # 念押しで前後の空白を削除する
    text = text.strip()

    # 連続する句読点を1つにまとめる
    if is_japanese is True:
        text = re.sub(r'([、。！？])\1+', r'\1', text)
    else:
        text = re.sub(r'([,\.!\?])\1+', r'\1', text)

    return text
