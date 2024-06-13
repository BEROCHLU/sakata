#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import datetime
import os

import pandas as pd
import requests
from dateutil import tz

# hash
str_k = b"aHR0cHM6Ly9rYWJ1dGFuLmpwL3N0b2NrL2thYnVrYT9jb2RlPTEzMjEmYXNoaT1kYXkmcGFnZT0="
str_m = b"aHR0cHM6Ly9meC5taW5rYWJ1LmpwL2FwaS92Mi9iYXIvVVNESlBZL2RhaWx5Lmpzb24="
str_ua = b"TW96aWxsYS81LjAgKE1hY2ludG9zaDsgSW50ZWwgTWFjIE9TIFggMTBfMTVfNykgQXBwbGVXZWJLaXQvNTM3LjM2IChLSFRNTCwgbGlrZSBHZWNrbykgQ2hyb21lLzEyNS4wLjAuMCBTYWZhcmkvNTM3LjM="
# path
output_folder = "./csv"
# set timezone
edt = tz.gettz("America/New_York")
# lambda
f1 = lambda dt: dt + datetime.timedelta(days=-3) if dt.weekday() == 0 else dt + datetime.timedelta(days=-1)
f2 = lambda ms: datetime.datetime.fromtimestamp(ms, tz=edt).strftime("%Y-%m-%d")
f3 = lambda ns: datetime.datetime.fromtimestamp(ns / 1000).strftime("%Y-%m-%d")  # timestanpがおかしい24H足りない JSTに変換した日付をEDTの日付とみなすと正しい


def getDataFrame1():
    lst_page = []
    lst_df = None
    page = base64.b64decode(str_k).decode()

    for i in range(3):
        url = f"{page}{i+1}"
        lst_df = pd.read_html(url, header=0, index_col=0)
        df_page = lst_df[5]
        lst_page.append(df_page)

    lst_df[4].index.name = "日付"
    lst_page.append(lst_df[4])

    df_concat = pd.concat(lst_page)  # page結合
    df_concat.sort_values(by="日付", inplace=True)  # 下が最新になるようにソート
    df_concat.reset_index(inplace=True)  # 日付がindexになってるので振り直し

    df_concat["日付"] = pd.to_datetime(df_concat["日付"], format="%y/%m/%d")  # 日付のフォーマット指定変換 yy/mm/dd => yyyy-mm-dd
    df_concat["日付"] = df_concat["日付"].map(f1)  # 月曜日だったら先週の金曜日、それ以外は前日
    df_concat["日付"] = df_concat["日付"].dt.strftime("%Y-%m-%d")  # キャスト datetime64 to string

    print("Done NK")
    return df_concat


def getDataFrame2():
    lst_page = []
    lst_df = None
    page = "https://us.kabutan.jp/stocks/DIA/historical_prices/daily?page="

    for i in range(3):
        url = f"{page}{i+1}"
        lst_df = pd.read_html(url, header=0, index_col=0)
        df_page = lst_df[1]
        lst_page.append(df_page)

    lst_df[0].index.name = "日付"
    lst_page.append(lst_df[0])

    df_concat = pd.concat(lst_page)  # page結合
    df_concat.sort_values(by="日付", inplace=True)  # 下が最新になるようにソート
    df_concat.reset_index(inplace=True)  # 日付がindexになってるので振り直し

    df_concat["日付"] = pd.to_datetime(df_concat["日付"], format="%y/%m/%d")  # 日付のフォーマット指定変換 yy/mm/dd => yyyy-mm-dd
    df_concat["日付"] = df_concat["日付"].dt.strftime("%Y-%m-%d")  # キャスト datetime64 to string

    print("Done DIA")
    return df_concat
    '''
    ticker = "^DJI" # ^DJI | ^IXIC
    strRange = "6mo"

    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
    ua = base64.b64decode(str_ua).decode()

    data_chart = requests.get(url, params={"range": strRange, "interval": "1d"}, headers={"User-Agent": ua})
    data_chart = data_chart.json()

    hshResult = data_chart["chart"]["result"][0]
    hshQuote = hshResult["indicators"]["quote"][0]
    hshQuote["date"] = hshResult["timestamp"]

    df_quote = pd.DataFrame(hshQuote.values(), index=hshQuote.keys()).T
    df_quote = df_quote.dropna(subset=["open", "high", "low", "close"])  # OHLCに欠損値''が1つでもあれば行削除
    df_quote = df_quote.round(2)  # float64 => float32

    df_quote["date"] = df_quote["date"].map(f2)  # UNIX time to EDT Datetime string
    df_quote = df_quote.reindex(columns=["date", "open", "high", "low", "close", "volume"])  # sort columns

    print(f"Done {ticker}")
    return df_quote
    '''


def getDataFrame3():
    url = base64.b64decode(str_m).decode()
    data_eusd = requests.get(url, params={"count": 128})
    data_eusd = data_eusd.json()

    df_eusd = pd.DataFrame(data_eusd, columns=["date", "open", "high", "low", "close"])  # list to dataframe
    df_eusd["date"] = df_eusd["date"].map(f3)  # UNIX time to JST Datetime string
    df_eusd.drop(columns=["open", "high", "low"], inplace=True)  # いらない列削除
    df_eusd["close"] = pd.to_numeric(df_eusd["close"], errors="coerce").round(4)  # string to round float64

    print("Done Currency")
    return df_eusd


if __name__ == "__main__":
    # vlookup
    [df_concat, df_quote, df_eusd] = [getDataFrame1(), getDataFrame2(), getDataFrame3()]

    df_merge = pd.merge(df_concat, df_quote, on="日付")  # dateをstringで統一することでマージしやすくなる
    df_merge = pd.merge(df_merge, df_eusd, left_on="日付", right_on="date")
    df_merge = df_merge[["日付", "始値_x", "終値_y", "close"]]
    df_merge.rename(columns={"日付": "date", "始値_x": "open_t", "終値_y": "close_x", "close": "close_y"}, inplace=True)

    if not len(df_merge) > 80:
        print(f"warning: {len(df_merge)}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # サブフォルダも同時に再帰的に作成

    # download_path = os.path.join(output_folder, "datexyt.csv")
    # df_merge.to_csv(download_path, header=False, index=False, lineterminator="\n")  # for C
    download_path = os.path.join(output_folder, "hdatexyt.csv")
    df_merge.to_csv(download_path, header=True, index=False, lineterminator="\n")  # for Python, Node.js
