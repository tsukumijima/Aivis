
# Aivis

💠 **Aivis:** **AI** **V**oice **I**mitation **S**ystem

## Installation

Linux (Ubuntu 20.04 LTS) x64 でのみ検証しています。  
Windows 上での動作は想定していません。Windows では WSL2 を使ってください (未検証だがおそらく動くはず) 。

### non-Docker

Docker を使わない場合、事前に Python 3.11・Poetry・FFmpeg がインストールされている必要があります。

```bash
# サブモジュールが含まれているため --recurse を付ける
git clone --recurse https://github.com/tsukumijima/Aivis.git

# 依存関係のインストール
cd Aivis
poetry env use 3.11
poetry install --no-root

# ヘルプを表示
./Aivis.sh --help
```

### Docker

Docker を使う場合、事前に Docker がインストールされている必要があります。  
Docker を使わない場合と比べてあまり検証できていないため、うまく動かないことがあるかもしれません。


```bash
# サブモジュールが含まれているため --recurse を付ける
git clone --recurse https://github.com/tsukumijima/Aivis.git

# 依存関係のインストール
cd Aivis
./Aivis-Docker.sh build

# ヘルプを表示
./Aivis-Docker.sh --help
```

## Dataset Directory Structure

Aivis のデータセットディレクトリは、4段階に分けて構成されています。

- **01-Sources:** データセットにする音声をそのまま入れるディレクトリ
  - データセットの素材にする音声ファイルをそのまま入れてください。
    - 基本どの音声フォーマットでも大丈夫です。`create-segments` での下処理にて、自動的に wav に変換されます。
    - 背景 BGM の除去などの下処理を行う必要はありません。`create-segments` での下処理にて、自動的に BGM や雑音の除去が行われます。
    - 数十分〜数時間ある音声ファイルの場合は `create-segments` での書き起こしの精度が悪くなることがあるため、事前に10分前後に分割することをおすすめします。
  - `create-segments` サブコマンドを実行すると、BGM や雑音の除去・書き起こし・一文ごとのセグメントへの分割・セグメント化した音声の音量/フォーマット調整が、すべて自動的に行われます。
- **02-PreparedSources:** `create-segments` サブコマンドで下処理が行われた音声ファイルと、その書き起こしデータが入るディレクトリ
  - `create-segments` サブコマンドを実行すると、`01-Sources` にある音声ファイルの BGM や雑音が除去され、このディレクトリに書き起こしデータとともに保存されます。
  - `create-segments` の実行時、このディレクトリに当該音声の下処理済みの音声ファイルや書き起こしデータが存在する場合は、そのファイルが再利用されます。
  - 下処理済みの音声ファイル名は `02-PreparedSources/(01-Source でのファイル名).wav` となります。
  - 書き起こしデータのファイル名は `02-PreparedSources/(01-Source でのファイル名).json` となります。
    - 書き起こしの精度がよくない (Whisper のガチャで外れを引いた) 場合は、書き起こしデータの JSON ファイルを削除してから `create-segments` を実行すると、再度書き起こしが行われます。
- **03-Segments:** `create-segments` サブコマンドでセグメント化された音声ファイルが入るディレクトリ
  - `create-segments` サブコマンドを実行すると、`02-PreparedSources` にある音声ファイルが書き起こし分や無音区間などをもとに大まかに一文ごとに分割され、このディレクトリに保存されます。
  - セグメントデータのファイル名は `03-Segments/(01-Source でのファイル名)/(4桁の連番)_(書き起こし文).wav` となります。
  - なんらかの理由でもう一度セグメント化を行いたい場合は、`03-Segments/(01-Source でのファイル名)/` を削除してから `create-segments` を実行すると、再度セグメント化が行われます。
- **04-Datasets:** `create-datasets` サブコマンドで手動で作成されたデータセットが入るディレクトリ
  - `create-datasets` サブコマンドを実行すると Gradio の Web UI が起動し、`03-Segments` 以下にある一文ごとにセグメント化された音声と書き起こし文をもとにアノテーションを行い、手動でデータセットを作成できます。
  - `03-Segments` までの処理は AI 技術を使い完全に自動化されています。
    - 調整を重ねそれなりに高い精度で自動生成できるようになった一方で、他の人と声が被っていたり発音がはっきりしないなど、データセットにするにはふさわしくない音声が含まれていることもあります。
    - また、書き起こし文が微妙に誤っていたり、句読点がなかったりすることもあります。
    - さらに元の音声に複数の話者の声が含まれている場合、必要な話者の音声だけを抽出する必要もあります。
  - `create-datasets` サブコマンドで起動する Web UI は、どうしても最後は人力で行う必要があるアノテーション作業を、簡単に手早く行えるようにするためのものです。
    - 話者の選別 (データセットから除外することも可能)・音声の再生・音声のトリミング (切り出し)・書き起こし文の修正を一つの画面で行えます。
    - 確定ボタンを押すと、そのセグメントが指定された話者のデータセットに追加されます (データセットからの除外が指定された場合はスキップされる) 。
    - `create-datasets` サブコマンドによって、`03-Segments` 以下のセグメント化された音声ファイルが変更されることはありません。
  - データセットは音声ファイルが `04-Datasets/(話者名)/audio/wavs/(連番).wav` に、書き起こし文が `04-Datasets/(話者名)/filelists/speaker.list` にそれぞれ保存されます。
    - このディレクトリ構造は Bert-VITS2 のデータセット構造に概ね準拠したものですが、config.json など一部のファイルやディレクトリは存在しません。
    - `train` サブコマンドを実行すると、指定された話者のデータセットディレクトリが Bert-VITS2 側にコピーされ、別途 config.json など必要なファイルもコピーされた上で Bert-VITS2 の学習処理が開始されます。
    - Bert-VITS2 の学習処理によって、`04-Datasets` 以下のデータセットが変更されることはありません。

## License

[MIT License](License.txt)
