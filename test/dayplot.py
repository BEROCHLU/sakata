import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

# サンプルデータの生成
start_date = datetime(2023, 4, 1)
end_date = datetime(2023, 4, 30)
date_list = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
date_list = [date for date in date_list if date.weekday() < 5]  # 土日を除外

np.random.seed(0)
data = np.random.rand(len(date_list))

# 日付をインデックスに変換
dates_as_numbers = np.arange(len(date_list))

fig, ax = plt.subplots()
ax.plot(dates_as_numbers, data)  # インデックスを使用してプロット

# X軸のラベルを日付に設定
ax.set_xticks(dates_as_numbers)
ax.set_xticklabels([date.strftime("%b%d") for date in date_list], rotation=45, ha="right")

plt.title("Weekdays Data Plot without Gaps")
plt.xlabel("Date")
plt.ylabel("Value")
plt.grid()
plt.tight_layout()
plt.show()
