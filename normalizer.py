#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

DESIRED_ERROR = 0.001
PERIOD = 55


if __name__ == "__main__":
    in_path = "./json/hdatexyt.json"
    out_path = "./json/n225out.json"
    out_path2 = "./json/setting.json"

    with open(in_path, encoding="utf-8") as f:
        arrHsh = json.load(f)

    arrDate = list(map(lambda hsh: hsh["date"], arrHsh))
    arrDOW = list(map(lambda hsh: hsh["close_x"], arrHsh))
    arrFX = list(map(lambda hsh: hsh["close_y"], arrHsh))
    arrNK = list(map(lambda hsh: hsh["open_t"], arrHsh))

    arrChange_DOW = []
    arrChange_FX = []
    arrChange_NK = []

    arrDate.pop(0)  # 初日除外
    LEN = len(arrHsh)

    for i in range(LEN):
        if 0 < i:  # 変化率なので初日除外
            f0 = (arrDOW[i] / arrDOW[i - 1]) * 100
            arrChange_DOW.append(f0)
            f1 = (arrFX[i] / arrFX[i - 1]) * 100
            arrChange_FX.append(f1)
            f2 = (arrNK[i] / arrNK[i - 1]) * 100
            arrChange_NK.append(f2)

    days = LEN - 1  # 変化率なので初日除外
    SKIP = days - PERIOD

    del arrDate[:SKIP]
    del arrChange_DOW[:SKIP]
    del arrChange_FX[:SKIP]
    del arrChange_NK[:SKIP]

    DIV_DOW = max(arrChange_DOW) * (1 + DESIRED_ERROR)
    DIV_FX = max(arrChange_FX) * (1 + DESIRED_ERROR)
    DIV_NK = max(arrChange_NK) * (1 + DESIRED_ERROR)

    arrTrainX = []
    arrTrainT = []
    # 最大値で正規化
    for i in range(PERIOD):
        x0 = arrChange_DOW[i] / DIV_DOW
        x1 = arrChange_FX[i] / DIV_FX

        arrTrainX.append([x0, x1])
        arrTrainT.append([arrChange_NK[i] / DIV_NK])

    arrTrainData = []

    for x, t, d in zip(arrTrainX, arrTrainT, arrDate):
        a = {"input": x, "output": t, "date": d}
        arrTrainData.append(a)

    with open(out_path, "w") as f:
        json.dump(arrTrainData, f, indent=4)
    with open(out_path2, "w") as f:
        hsh = {"DIV_T": DIV_NK}
        json.dump(hsh, f)
