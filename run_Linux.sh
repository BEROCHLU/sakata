#!/bin/bash

# 出力ファイルを事前にクリア
> ./result/inlinestd.log

# batchフォルダ内の全ての.jsonファイルに対してループ
for file in ./batch/*.json; do
    # ファイルの内容を.pyに渡し、結果を出力ファイルに追記
    python ./src/python/inlinestd.py < "$file" >> ./result/inlinestd.log
done
