import datetime
import json
import math
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
from glob import glob
from multiprocessing import Manager
from pprint import pprint

import matplotlib.pyplot as plt

# lambda
dsigmoid = lambda a: a * (1 - a)
frandWeight = lambda: 0.5
frandBias = lambda: -1
# global
[IN_NODE, HID_NODE, OUT_NODE] = [None, None, 1]
DAYS = None
ETA = 0.5
THRESHOLD = 500000


def sigmoid(a: float) -> float:
    if a < 0:
        return math.exp(a) / (1 + math.exp(a))
    else:
        return 1 / (1 + math.exp(-a))


def updateHidOut(n: int, hid: float, out: float, x: float, v: float, w: float):
    for i in range(HID_NODE):
        dot_h = 0
        for j in range(IN_NODE):
            dot_h += x[n][j] * v[i][j]
        hid[i] = sigmoid(dot_h)

    hid[HID_NODE - 1] = frandBias()

    for i in range(OUT_NODE):
        dot_o = 0
        for j in range(HID_NODE):
            dot_o += w[i][j] * hid[j]
        out[i] = sigmoid(dot_o)


def printResult(arrHsh: list, DIV_T: float, epoch: int, fError: float, t: float, hid: float, out: float, x: float, v: float, w: float) -> list:
    arrErate = []
    arrPrint = []
    acc_min = sys.maxsize
    acc_max = -sys.maxsize

    for i in range(DAYS):
        updateHidOut(i, hid, out, x, v, w)

        arrErate.append(100 * (t[i][0] - out[0]) / t[i][0])

        accumulate = reduce((lambda result, current: result + current), arrErate)

        undo_out = round(out[0] * DIV_T, 2)
        undo_teacher = round(t[i][0] * DIV_T, 2)

        pad_out = str(undo_out).rjust(6)
        pad_teacher = str(undo_teacher).rjust(6)
        pad_erate = str(round(arrErate[i], 2)).rjust(5)
        pad_acc = str(round(accumulate, 2)).rjust(5)

        acc_max = accumulate if acc_max < accumulate else acc_max
        acc_min = accumulate if accumulate < acc_min else acc_min

        s = f"{arrHsh[i]['date']} {pad_out} True: {pad_teacher} Err: {pad_erate} Acc: {pad_acc}"
        arrPrint.append(s)

    acc_mid = (acc_max + acc_min) / 2
    acc_nom = (accumulate - acc_min) * 100 / (acc_max - acc_min)

    lst_abs = list(map(lambda fErate: abs(fErate), arrErate))
    fMean = statistics.mean(lst_abs)
    # arrPlotAcc.append(acc_nom)  # plot
    s = f"Average error: {round(fMean, 2)}%"
    arrPrint.append(s)
    s = f"Min: {round(acc_min, 2)} Max: {round(acc_max, 2)} Mid: {round(acc_mid, 2)} Epoch: {epoch} Days: {DAYS}"
    arrPrint.append(s)
    s = f"Nom: {round(acc_nom, 2)} FinalErr: {round(fError, 5)}"
    arrPrint.append(s)

    return arrPrint


def addBias(hsh: dict) -> dict:
    arrInput = hsh["input"]
    arrInput.append(frandBias())  # add bias
    return arrInput


def main(strPath: str, lst_mg: list):
    global IN_NODE
    global HID_NODE
    global DAYS

    f = open(strPath, "r")
    dc_raw = json.load(f)
    arrHsh = dc_raw["listdc"]
    DIV_T = dc_raw["div"]

    x = list(map(addBias, arrHsh))
    t = list(map(lambda hsh: hsh["output"], arrHsh))

    IN_NODE = len(x[0])  # get input length include bias
    HID_NODE = IN_NODE + 1
    DAYS = len(x)

    hid = [0] * HID_NODE
    out = [0] * OUT_NODE
    delta_hid = [0] * HID_NODE
    delta_out = [0] * OUT_NODE
    epoch = 0
    v, w = [], []
    fError = 0

    for _ in range(HID_NODE):
        v.append([])
    for _ in range(OUT_NODE):
        w.append([])

    for i in range(HID_NODE):
        for j in range(IN_NODE):
            v[i].append(frandWeight())  # random() | uniform(0.5, 1.0)
    for i in range(OUT_NODE):
        for j in range(HID_NODE):
            w[i].append(frandWeight())  # random() | uniform(0.5, 1.0)

    while epoch < THRESHOLD:
        epoch += 1
        fError = 0.0

        for n in range(DAYS):
            updateHidOut(n, hid, out, x, v, w)

            for k in range(OUT_NODE):
                fError += 0.5 * (t[n][k] - out[k]) ** 2
                delta_out[k] = (t[n][k] - out[k]) * out[k] * (1 - out[k])

            for k in range(OUT_NODE):
                for j in range(HID_NODE):
                    w[k][j] += ETA * delta_out[k] * hid[j]

            for i in range(HID_NODE):
                delta_hid[i] = 0

                for k in range(OUT_NODE):
                    delta_hid[i] += delta_out[k] * w[k][i]

                delta_hid[i] = dsigmoid(hid[i]) * delta_hid[i]

            for i in range(HID_NODE):
                for j in range(IN_NODE):
                    v[i][j] += ETA * delta_hid[i] * x[n][j]
        # for days
        if (epoch % 100) == 0:
            lst_mg.append(fError)
            pass
    # while
    return printResult(arrHsh, DIV_T, epoch, fError, t, hid, out, x, v, w)


if __name__ == "__main__":
    # プロットのため各プロセスとメモリ共有
    lst_mg = Manager().list()
    lst_plot = Manager().list()

    timeStart = time.time()
    date_now = datetime.datetime.now()
    print(date_now.strftime("%F %T"))

    DIR_PATH = "batch"
    lst_strPath = glob(f"{DIR_PATH}/*.json")
    arrPrint = []

    lst_g = [lst_mg for _ in range(len(lst_strPath))]
    """
    excuter = ProcessPoolExecutor(max_workers=4)
    for (i, strPath) in enumerate(lst_strPath):
        if not (16 <= i and i <= sys.maxsize):  # pass loop index
            continue
        excuter.submit(main, strPath)
        #main(strPath)
    excuter.shutdown(wait=True)
    """
    # shutdown不要
    with ProcessPoolExecutor(max_workers=4) as excuter:
        arrPrint = list(excuter.map(main, lst_strPath[0:], lst_g))

    pprint(arrPrint)
    # measure time
    timeEnd = time.time()
    nSec = int(timeEnd - timeStart)
    nMinute = int(nSec / 60) if 60 <= nSec else 0
    print(f"Time: {nMinute} min {nSec % 60} sec.\n")
    # show plot
    plt.subplot(2, 1, 1)
    plt.plot(lst_mg)
    plt.subplot(2, 1, 2)
    plt.plot(lst_plot)
    plt.show()
