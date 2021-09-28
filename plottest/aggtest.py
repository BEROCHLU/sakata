import os
import matplotlib
from matplotlib import pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # カレントフォルダを実行ファイルパスに変更する

x = [1, 2, 3, 4, 5]
h = [2, 5, 3, 4, 1]
w = [0.8] * 5

print(matplotlib.get_backend())

matplotlib.use("agg")

fig, ax = plt.subplots()
ax.bar(x, h, width=w, align='edge')

fig.savefig('bar.png')
plt.show()
