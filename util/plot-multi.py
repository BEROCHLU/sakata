import re
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.ticker import AutoMinorLocator

file_paths = ["./result/main-batch.log", "./result/sakata-batch.log"]
norm_values_list = []
date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")

for file_path in file_paths:
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    sections = content.split("\n\n")
    dates = []
    norm_values = []

    for section in sections:
        lines = section.split("\n")
        lines = lines[1:]

        for line in reversed(lines):
            date_match = date_pattern.search(line)
            if date_match:
                date_str = date_match.group()
                date = datetime.strptime(date_str, "%Y-%m-%d").date()
                # dates.append(date)
                dates.append(date.strftime("%b%d"))
                break

        norm_line = [line for line in lines if line.startswith("Norm:")]
        if norm_line:
            norm_value = float(norm_line[0].split(":")[1].strip())
            norm_values.append(norm_value)

    norm_values_list.append(norm_values)

plt.figure(figsize=(14, 7))
for norm_values in norm_values_list:
    plt.plot(dates, norm_values, marker="o", markersize=4)

# X軸の範囲を調整
if dates:
    plt.xlim([dates[0], dates[-1]])

plt.gca().yaxis.set_minor_locator(AutoMinorLocator(2)) # Y軸の補助メモリを2ずつに設定
plt.title("The Sakata Index", fontsize=10)
plt.xticks(fontsize=9)  # X軸の目盛りのフォントサイズを8に設定
plt.grid(which="both")
plt.tight_layout()
plt.savefig("./result/plot-multi.png")  # showの前でないと保存されない
plt.show()
