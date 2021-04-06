import json

import pandas as pd

# global
old_ite = -1


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

    df_hdatexyt = pd.read_csv("./csv/hdatexyt.csv")
    skip_index = len(df_hdatexyt) - PERIOD
    lst_skip = [i for i in range(skip_index)]  # skipする日数リスト

    df_change["date"] = df_hdatexyt["date"]
    df_change["close_x"] = df_hdatexyt["close_x"].map(f1)  # 前日比%
    df_change["close_y"] = df_hdatexyt["close_y"].map(f1)  # 前日比%
    df_change["open_t"] = df_hdatexyt["open_t"].map(f1)  # 前日比%

    df_change.drop(index=lst_skip, inplace=True)
    df_change.reset_index(drop=True, inplace=True)  # dropしたのでindex振り直し いるか？

    df_normalize = df_change.drop(columns="date")  # dataframe全体に正規化を適用するのでdateを一時的に外す
    df_normalize = df_normalize / (df_normalize.max() * (1 + DESIRED_ERROR))
    df_normalize["date"] = df_change["date"]

    lst_close_x = df_normalize["close_x"].values.tolist()
    lst_close_y = df_normalize["close_y"].values.tolist()
    lst_open_t = df_normalize["open_t"].values.tolist()
    lst_date = df_normalize["date"].values.tolist()

    for x, y, t, d in zip(lst_close_x, lst_close_y, lst_open_t, lst_date):
        dc = {"input": [x, y], "output": [t], "date": d}
        lst_dc.append(dc)

    with open("./json/n225out.json", "w") as f:
        json.dump(lst_dc, f, indent=4)

    DIV_T = df_change["open_t"].max() * (1 + DESIRED_ERROR)  # 学習結果のアウトプットを正規化前に戻すため除数を渡す

    with open("./json/setting.json", "w") as f:
        hsh = {"DIV_T": DIV_T}
        json.dump(hsh, f)
