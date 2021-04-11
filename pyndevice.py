import datetime
import json
import math
import random
import statistics
import sys
import time
from functools import reduce

import matplotlib.pyplot as plt
import pandas as pd

# lambda
dsigmoid = lambda a: a * (1 - a)
frandWeight = lambda: 0.5
frandBias = lambda: -1
# global
THRESHOLD = 500000
OUT_NODE = 1
ETA = 0.5
IN_NODE, HID_NODE = None, None
hid, out = None, None
delta_hid, delta_out = None, None
epoch, days = 0, 0
fError = sys.maxsize
x, t = None, None
v, w = [], []
isPlot = True


def sigmoid(a: float) -> float:
    if a < 0:
        return math.exp(a) / (1 + math.exp(a))
    else:
        return 1 / (1 + math.exp(-a))


def updateHidOut(n: int):
    for i in range(HID_NODE):
        dot_h = 0
        for j in range(IN_NODE):
            dot_h += x[n][j] * v[i][j]
        hid[i] = sigmoid(dot_h)

    hid[HID_NODE - 1] = frandBias()  # random.random()

    for i in range(OUT_NODE):
        dot_o = 0
        for j in range(HID_NODE):
            dot_o += w[i][j] * hid[j]
        out[i] = sigmoid(dot_o)


def printResult(DIV_T: float):
    arrErate = []
    acc_min = sys.maxsize
    acc_max = -sys.maxsize

    for i in range(days):

        updateHidOut(i)

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

        print(f"{arrHsh[i]['date']} {pad_out} True: {pad_teacher} Err: {pad_erate} Acc: {pad_acc}")

    acc_mid = (acc_max + acc_min) / 2
    acc_nom = (accumulate - acc_min) * 100 / (acc_max - acc_min)

    f_time = time_ed - time_st

    if 60 <= f_time:
        n_minute = int(f_time / 60)
        f_sec = round(f_time % 60)
    else:
        n_minute = 0
        f_sec = round(f_time, 2)

    lst_abs = list(map(lambda fErate: abs(fErate), arrErate))
    fMean = statistics.mean(lst_abs)

    print(f"Average error: {round(fMean, 2)}%")
    print(f"Min: {round(acc_min, 2)} Max: {round(acc_max, 2)} Mid: {round(acc_mid, 2)}")
    print(f"Epoch: {epoch} Days: {days}")
    print(f"Nom: {round(acc_nom, 2)}")
    print(f"Time: {n_minute} min {f_sec} sec. FinalErr: {round(fError, 5)}\n")


def addBias(hsh: dict) -> dict:
    arrInput = hsh["input"]
    arrInput.append(frandBias())  # add bias
    return arrInput


if __name__ == "__main__":
    f = open("./json/seikika.json", "r")  # xor | cell30
    dc_raw = json.load(f)
    arrHsh = dc_raw["listdc"]
    DIV_T = dc_raw["div"]

    x = list(map(addBias, arrHsh))
    t = list(map(lambda hsh: hsh["output"], arrHsh))

    IN_NODE = len(x[0])  # get input length include bias
    HID_NODE = IN_NODE + 1
    days = len(x)

    hid = [0] * HID_NODE
    out = [0] * OUT_NODE
    delta_hid = [0] * HID_NODE
    delta_out = [0] * OUT_NODE

    arrErr = []

    for i in range(HID_NODE):
        v.append([])
    for i in range(OUT_NODE):
        w.append([])

    for i in range(HID_NODE):
        for j in range(IN_NODE):
            v[i].append(frandWeight())  # random() | uniform(0.5, 1.0)
    for i in range(OUT_NODE):
        for j in range(HID_NODE):
            w[i].append(frandWeight())  # random() | uniform(0.5, 1.0)

    date_now = datetime.datetime.now()
    print(date_now.strftime("%F %T"))

    time_st = time.time()

    while epoch < THRESHOLD:
        epoch += 1
        fError = 0

        for n in range(days):
            updateHidOut(n)

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
        if isPlot:
            if epoch % 100 == 0:
                # print(f"{epoch}: {fError}")
                arrErr.append(fError)
    # while
    time_ed = time.time()
    printResult(DIV_T)
    # show plot
    if isPlot:
        plt.plot(arrErr)
        plt.show()
