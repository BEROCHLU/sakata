import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.ticker import AutoMinorLocator


# ファイルパス
file_path = "./result/main.txt"

with open(file_path, "r", encoding="utf-8") as file:
    # ファイルの内容を1行ずつ読み込む
    lstStrContents = file.readlines()

# セクションごとの最後の日付とNormの値を抽出
lstDt = []
lstAcc = []

# 日付のパターン
date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")

for strContents in lstStrContents:
    date_match = date_pattern.search(strContents)
    if date_match:
        strLine = date_match.string.split("\n")[0] # delete \n
        lst_split = strLine.split(" ")
        date = datetime.strptime(lst_split[0], "%Y-%m-%d") # convert string to datetime
        lstDt.append(date)
        lstAcc.append(float(lst_split[-1]))
        pass

# グラフを描画
# 日付を0から始まる連続したインデックスに変換
arrInt = np.arange(len(lstDt))

fig, ax = plt.subplots(figsize=(12, 6))

# インデックスを使用してプロット
ax.plot(arrInt, lstAcc, marker="o", markersize=4)
# X軸のラベルをインデックスに設定
ax.set_xticks(arrInt)
# インデックスをstring日付に置き換える
ax.set_xticklabels([date.strftime("%b%d") for date in lstDt], rotation=45, ha="right")
# X軸の範囲を調整
ax.set_xlim([0, len(lstDt) - 1])

# plt.gcf().autofmt_xdate()  # X軸の日付ラベルを斜めにして重なりを防ぐ

plt.gca().yaxis.set_minor_locator(AutoMinorLocator(2))  # Y軸の補助メモリを2ずつに設定
plt.title("acc", fontsize=10)
plt.xticks(fontsize=9)  # X軸の目盛りのフォントサイズを設定
plt.grid(which="both")
plt.tight_layout()
plt.savefig("./result/latest-acc.png")  # showの前でないと機能しない
plt.show()
