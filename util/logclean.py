#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

# path
RW_PATH = "../result/pbatch.log"


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 実行ファイルパスをカレントフォルダに変更する
    strRead = ""
    with open(RW_PATH, "r") as f:
        strRead = f.read()

    with open(RW_PATH, "w") as f:
        s = re.sub("[\[\]',]", "", strRead)
        f.write(s)
