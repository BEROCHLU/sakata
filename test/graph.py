import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# (日付, float) 形式のタプルのリスト
data = [("2023-01-01", 1.0), ("2023-02-01", 3.5), ("2023-03-01", 2.4)]

# 日付と値を分けてリストに格納
# dates = [datetime.strptime(date, "%Y-%m-%d") for date, value in data]
# values = [value for date, value in data]

# プロット
plt.figure(figsize=(10, 6))  # グラフのサイズ設定
# plt.plot(dates, values, marker="o")  # 日付と値をプロット
plt.plot(*zip(*data), marker="o")  # 日付と値をプロット



# 日付フォーマットの設定
#plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
#plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

# 自動的に日付ラベルを斜めにする
#plt.gcf().autofmt_xdate()

plt.show()
