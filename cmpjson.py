#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

if __name__ == "__main__":
    path1 = "./json/py225.json"
    path2 = "./json/n225out.json"
    with open(path1, encoding="utf-8") as f:
        arrHsh1 = json.load(f)
    with open(path2, encoding="utf-8") as f:
        arrHsh2 = json.load(f)

    if arrHsh1 == arrHsh2:
        print(True)
    else:
        print(False)
