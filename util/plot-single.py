import re
import datetime
import matplotlib.pyplot as plt

# ファイルを読み込む
file_path = "./result/main-batch.log"

with open(file_path, "r", encoding="utf-8") as file:
    # ファイルの内容を読み込む
    content = file.read()

# セクションを分割
sections = content.split("\n\n")

# セクションごとの最後の日付とNormの値を抽出
dates = []
norm_values = []

# 日付のパターン
date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")

for section in sections:
    lines = section.split("\n")
    # 最初の行はタイムスタンプなのでスキップ
    lines = lines[1:]

    for line in reversed(lines):
        # 日付を検索
        date_match = date_pattern.search(line)
        if date_match:
            # 日付を解析しリストに追加
            date_str = date_match.group()
            date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            # dates.append(date_str)
            dates.append(date.strftime("%b%d"))
            break

    # Normの値を検索
    norm_line = [line for line in lines if line.startswith("Norm:")]
    if norm_line:
        norm_value = float(norm_line[0].split(":")[1].strip())
        norm_values.append(norm_value)

# グラフを描画
plt.figure(figsize=(15, 8))
plt.plot(dates, norm_values, marker="o")
plt.title("The Sakata Index", fontsize=10)
plt.xticks(fontsize=8)  # X軸の目盛りのフォントサイズを8に設定
plt.grid()
plt.tight_layout()
plt.savefig("./result/plot-single.png")  # showの前でないと機能しない
plt.show()
