import json
import math
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import reduce
from glob import glob
from multiprocessing import Manager
from pprint import pprint
from time import time

import matplotlib.pyplot as plt

# lambda
frandWeight = lambda: 0.5
frandBias = lambda: -1
# global
[IN_NODE, HID_NODE, OUT_NODE] = [None, None, 1]
DAYS = None
ETA = 0.5
THRESHOLD = 500000


def addBias(hsh: dict) -> dict:
    arrInput = hsh["input"]
    arrInput.append(frandBias())  # add bias
    return arrInput


def main(strPath, lst_c0, lst_c1):
    f = open(strPath, "r")
    dc_raw = json.load(f)
    arrHsh = dc_raw["listdc"]
    DIV_T = dc_raw["div"]

    x = list(map(addBias, arrHsh))
    t = list(map(lambda hsh: hsh["output"], arrHsh))

    onn = ONN(x)  # create instance
    arrPrint = onn.train(x, t, DIV_T, arrHsh, lst_c0, lst_c1)  # call train method
    return arrPrint


class ONN:  # Out of date Neural Network
    def __init__(self, x):
        global IN_NODE
        global HID_NODE
        global DAYS

        IN_NODE = len(x[0])  # get input length include bias
        HID_NODE = IN_NODE + 1
        DAYS = len(x)

    def sigmoid(self, a: float) -> float:
        if a < 0:
            return math.exp(a) / (1 + math.exp(a))
        else:
            return 1 / (1 + math.exp(-a))

    def calculateNode(self, n, x, v, w, hid, out):
        for i in range(HID_NODE):
            dot_h = 0.0
            for j in range(IN_NODE):
                dot_h += x[n][j] * v[i][j]
            hid[i] = self.sigmoid(dot_h)

        hid[HID_NODE - 1] = frandBias()

        for i in range(OUT_NODE):
            dot_o = 0.0
            for j in range(HID_NODE):
                dot_o += hid[j] * w[i][j]
            out[i] = self.sigmoid(dot_o)

    def printResult(self, arrHsh, DIV_T, epoch, fError, t, hid, out, x, v, w, lst_c1):
        arrErate = []
        arrPrint = []
        acc_min = sys.maxsize
        acc_max = -sys.maxsize
        accumulate = 0

        for i in range(DAYS):
            self.calculateNode(i, x, v, w, hid, out)  # 最後に更新された重みでノードを計算

            arrErate.append(100 * (t[i][0] - out[0]) / t[i][0])

            acc_max = accumulate if acc_max < accumulate else acc_max
            acc_min = accumulate if accumulate < acc_min else acc_min

            accumulate = reduce((lambda result, current: result + current), arrErate)

            undo_out = round(out[0] * DIV_T, 2)
            undo_teacher = round(t[i][0] * DIV_T, 2)

            pad_out = str(undo_out).rjust(6)
            pad_teacher = str(undo_teacher).rjust(6)
            pad_erate = str(round(arrErate[i], 2)).rjust(5)
            pad_acc = str(round(accumulate, 2)).rjust(5)

            arrPrint.append(f"{arrHsh[i]['date']} {pad_out} True: {pad_teacher} {pad_erate}% {pad_acc}")

        acc_mid = (acc_max + acc_min) / 2
        acc_nom = (accumulate - acc_min) * 100 / (acc_max - acc_min)

        lst_abs = list(map(lambda fErate: abs(fErate), arrErate))
        fMean = statistics.mean(lst_abs)
        lst_c1.append(acc_nom)  # plot
        arrPrint.append(f"Average error: {round(fMean, 2)}%")
        arrPrint.append(f"Min: {round(acc_min, 2)} Max: {round(acc_max, 2)} Mid: {round(acc_mid, 2)} Epoch: {epoch} Days: {DAYS}")
        arrPrint.append(f"Nom: {round(acc_nom, 2)} FinalErr: {round(fError, 5)}")
        arrPrint.append("")

        return arrPrint

    # training
    def train(self, x, t, DIV_T, arrHsh, lst_c0, lst_c1):
        hid = [0] * HID_NODE
        out = [0] * OUT_NODE
        delta_hid = [0] * HID_NODE
        delta_out = [0] * OUT_NODE
        epoch = 0
        fError = 0.0
        v, w = [], []  # v: weight in-hid, w: weight hid-out

        for _ in range(HID_NODE):
            v.append([])
        for _ in range(OUT_NODE):
            w.append([])

        for i in range(HID_NODE):
            for _ in range(IN_NODE):
                v[i].append(frandWeight())
        for i in range(OUT_NODE):
            for _ in range(HID_NODE):
                w[i].append(frandWeight())

        while epoch < THRESHOLD:
            epoch += 1
            fError = 0.0

            for n in range(DAYS):
                self.calculateNode(n, x, v, w, hid, out)

                for i in range(OUT_NODE):  # Δw 修正量計算
                    fError += 0.5 * (t[n][i] - out[i]) ** 2  # 損失関数
                    delta_out[i] = (t[n][i] - out[i]) * out[i] * (1 - out[i])

                for i in range(OUT_NODE):  # Δw = ("t−o" )∙"o" ("1−o" )"∙h"
                    for j in range(HID_NODE):
                        w[i][j] += ETA * delta_out[i] * hid[j]  # 𝑤_𝑛𝑒𝑤 "=" 𝑤_old " − " η"∆w"

                for i in range(HID_NODE):  # Δv 修正量計算
                    delta_hid[i] = 0
                    for j in range(OUT_NODE):  # h(1−h)∙{∑(t−𝑜)𝑜(1−𝑜)𝑤}
                        delta_hid[i] += delta_out[j] * w[j][i]
                    delta_hid[i] = hid[i] * (1 - hid[i]) * delta_hid[i]

                for i in range(HID_NODE):  # Δv = h(1−h)∙{∑(t−𝑜)𝑜(1−𝑜)𝑤}∙x
                    for j in range(IN_NODE):
                        v[i][j] += ETA * delta_hid[i] * x[n][j]  # 𝑣_𝑛𝑒𝑤 "=" 𝑣_old " − " η"∆v"
            # end for DAYS
            if (epoch % 100) == 0:
                lst_c0.append(fError)
                pass
        # end while
        return self.printResult(arrHsh, DIV_T, epoch, fError, t, hid, out, x, v, w, lst_c1)


if __name__ == "__main__":
    # プロットのため各プロセスとメモリ共有
    lst_mg0 = Manager().list()
    lst_mg1 = Manager().list()

    TIME_START = time()
    date_now = datetime.now()
    print(date_now.strftime("%F %T"))

    DIR_PATH = "batch"
    lst_strPath = glob(f"{DIR_PATH}/*.json")
    arrPrint = []

    lst_c0 = [lst_mg0 for _ in range(len(lst_strPath))]
    lst_c1 = [lst_mg1 for _ in range(len(lst_strPath))]
    # shutdown不要
    with ProcessPoolExecutor(max_workers=4) as excuter:
        arrPrint = list(excuter.map(main, lst_strPath[:], lst_c0, lst_c1))

    pprint(arrPrint)
    # measure time
    TIME_END = time()
    INT_SEC = int(TIME_END - TIME_START)
    INT_MINUTE = int(INT_SEC / 60) if 60 <= INT_SEC else 0
    print(f"Time: {INT_MINUTE} min {INT_SEC % 60} sec.\n")
    # show plot
    plt.subplot(2, 1, 1)
    plt.plot(lst_mg0)
    plt.subplot(2, 1, 2)
    plt.plot(lst_mg1)
    plt.show()
