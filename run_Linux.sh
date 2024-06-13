#!/bin/bash

# バッチファイル生成
python ./utils/slicebatch.py

# 出力ファイルを事前にクリア
> ./result/output3.log

# batchフォルダ内の全ての.jsonファイルに対してループ
for file in ./batch/*.json; do
    # ファイルの内容を.pyに渡し、結果を出力ファイルに追記
    python ./src/python/inlinestd.py < "$file" >> ./result/output3.log
done

# 画像出力
python ./src/python/plot-batch.py ./result/output3.log