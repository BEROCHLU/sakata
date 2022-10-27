#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

import pandas as pd

# global
old_ite = -1
# path
R_PATH = "../csv/hdatexyt.csv"
W_PATH = "../json/seikika.json"


def f1(ite):
    global old_ite  # グローバル変数を更新するため宣言が必要
    f_change = ite / old_ite * 100
    old_ite = ite
    return f_change


if __name__ == "__main__":
    DESIRED_ERROR = 0.001
    PERIOD = 55  # PERIOD以下であった場合のエラー処理
    df_change = pd.DataFrame()
    lst_dc = []

    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 実行ファイルパスをカレントフォルダに変更する
    df_hdatexyt = pd.read_csv(R_PATH)

    df_change["date"] = df_hdatexyt["date"]
    df_change["close_x"] = df_hdatexyt["close_x"].map(f1)  # 前日比%
    df_change["close_y"] = df_hdatexyt["close_y"].map(f1)  # 前日比%
    df_change["close"] = df_hdatexyt["close"].map(f1)  # 前日比%

    df_change = df_change.tail(PERIOD)  # tailからPERIODまで抽出

    df_normalize = df_change.drop(columns="date")  # dataframe全体に正規化を適用するのでdateを一時的に外す
    df_normalize = df_normalize / (df_normalize.max() * (1 + DESIRED_ERROR))
    df_normalize["date"] = df_change["date"]

    lst_close_x = df_normalize["close_x"].values.tolist()
    lst_close_y = df_normalize["close_y"].values.tolist()
    lst_open_t = df_normalize["close"].values.tolist()
    lst_date = df_normalize["date"].values.tolist()

    for x, y, t, d in zip(lst_close_x, lst_close_y, lst_open_t, lst_date):
        dc = {"input": [x, y], "output": [t], "date": d}
        lst_dc.append(dc)

    DIV_NK = df_change["close"].max() * (1 + DESIRED_ERROR)  # 学習結果のアウトプットを正規化前に戻すため除数を渡す
    dc_seikika = {"listdc": lst_dc, "div": DIV_NK}

    with open(W_PATH, "w") as f:
        json.dump(dc_seikika, f, indent=4)
    
    print("Done Normalize")
