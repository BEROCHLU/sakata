#!/bin/bash

# ログファイルのパスを変数として定義
LOG_FILE=./result/output3.log

# バッチファイル生成
python ./utils/slicebatch.py

# 開始時間の記録
start_time=$(date +%s)

# 出力ファイルを事前にクリアして現在の日付と時刻を追記
date > "$LOG_FILE"

# batchフォルダ内の全ての.jsonファイルに対してループ
for file in ./batch/*.json; do
    # ファイルの内容を.pyに渡し、結果を出力ファイルに追記
    python ./src/python/inlinestd.py < "$file" >> "$LOG_FILE"
done

# 画像出力
python ./src/python/plot-batch.py "$LOG_FILE"

# 終了時間の記録
end_time=$(date +%s)

# 経過時間の計算（秒単位）
elapsed_time=$((end_time - start_time))

# 経過時間を分と秒に変換
minutes=$((elapsed_time / 60))
seconds=$((elapsed_time % 60))

# 経過時間をログファイルに追記
echo "Time: $minutes min :$seconds sec." >> "$LOG_FILE"
