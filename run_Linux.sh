#!/bin/bash

# ログファイルのパスを変数として定義
LOG_FILE=./result/output3.log

# バッチファイル生成
python ./utils/slicebatch.py

# 出力ファイルを事前にクリアして現在の日付と時刻を追記
date > "$LOG_FILE"

# batchフォルダ内の全ての.jsonファイルに対してループ
for file in ./batch/*.json; do
    # ファイルの内容を.pyに渡し、結果を出力ファイルに追記
    python ./src/python/inlinestd.py < "$file" >> "$LOG_FILE"
done

# 画像出力
python ./src/python/plot-batch.py "$LOG_FILE"