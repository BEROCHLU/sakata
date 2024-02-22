import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import matplotlib.dates as mdates

# サンプルデータの生成
start_date = datetime(2023, 4, 1)
end_date = datetime(2023, 4, 30)
date_list = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
date_list = [date for date in date_list if date.weekday() < 5]  # 土日を除外

np.random.seed(0)
data = np.random.rand(len(date_list))

# 日付を連続したインデックスに変換
dates_as_numbers = np.arange(len(date_list))
#dates_as_numbers =  mdates.date2num(date_list)

fig, ax = plt.subplots()
ax.plot(dates_as_numbers, data)  # インデックスを使用してプロット

# X軸のラベルを日付に設定
ax.set_xticks(dates_as_numbers)
# インデックスをstring日付に置き換える
ax.set_xticklabels([date.strftime("%b%d") for date in date_list], rotation=45, ha="right")
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

#plt.gcf().autofmt_xdate()

plt.title("Weekdays Data Plot without Gaps")
plt.xlabel("Date")
plt.ylabel("Value")
plt.tight_layout()
plt.show()
