#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from glob import glob

import pandas as pd

# global
old_ite = -1
# path
R_PATH = "../csv/hdatexyt.csv"
BATCH_PATH = "../batch"


def f1(ite):
    global old_ite  # グローバル変数を更新するため宣言が必要
    f_change = ite / old_ite * 100
    old_ite = ite
    return f_change


if __name__ == "__main__":
    BATCH_SIZE = 55
    DESIRED_ERROR = 0.001

    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 実行ファイルパスをカレントフォルダに変更する
    df_hdatexyt = pd.read_csv(R_PATH)
    df_change = pd.DataFrame()

    df_change["date"] = df_hdatexyt["date"]
    df_change["close_x"] = df_hdatexyt["close_x"].map(f1)  # 前日比%
    df_change["close_y"] = df_hdatexyt["close_y"].map(f1)  # 前日比%
    df_change["close"] = df_hdatexyt["close"].map(f1)  # 前日比%

    df_change.drop(index=0, inplace=True)  # 変化率なので初日除外
    df_change.reset_index(drop=True, inplace=True)  # for iループのためインデックス振り直し

    if not os.path.exists(BATCH_PATH):
        os.mkdir(BATCH_PATH)  # フォルダ新規作成
    else:
        for file in glob(f"{BATCH_PATH}/*.json"):
            os.remove(file)  # 古いファイル削除

    FILE_KAZU = len(df_change) - BATCH_SIZE + 1  # スライスするファイル数
    LAST_INDEX = BATCH_SIZE - 1

    for i in range(FILE_KAZU):
        df_batch = df_change.loc[i : LAST_INDEX + i]

        df_normalize = df_batch.drop(columns="date")  # dataframe全体に正規化を適用するのでdateを一時的に外す
        df_normalize = df_normalize / (df_normalize.max() * (1 + DESIRED_ERROR))
        df_normalize["date"] = df_batch["date"]

        lst_close_x = df_normalize["close_x"].values.tolist()
        lst_close_y = df_normalize["close_y"].values.tolist()
        lst_open_t = df_normalize["close"].values.tolist()
        lst_date = df_normalize["date"].values.tolist()

        lst_dc = []

        for x, y, t, d in zip(lst_close_x, lst_close_y, lst_open_t, lst_date):
            dc = {"input": [x, y], "output": [t], "date": d}
            lst_dc.append(dc)

        DIV_NK = df_batch["close"].max() * (1 + DESIRED_ERROR)  # 学習結果のアウトプットを正規化前に戻すため除数を渡す
        dc_seikika = {"listdc": lst_dc, "div": DIV_NK}
        pad_z = str(i).zfill(2)

        with open(f"{BATCH_PATH}/seikika{pad_z}.json", "w") as f:
            json.dump(dc_seikika, f, indent=4)

    print("Done slice batch")
