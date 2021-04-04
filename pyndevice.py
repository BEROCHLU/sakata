#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import json
import math
import random
import time

import matplotlib.pyplot as plt
import pandas as pd

THRESHOLD = 500000
OUT_NODE = 1
ETA = 0.5


def sigmoid(a: float) -> float:
    if a < 0:
        return 1 - 1 / (1 + math.exp(a))
    else:
        return 1 / (1 + math.exp(-a))


dsigmoid = lambda a: a * (1 - a)
dmax = lambda a: a if (0 < a) == 1 else 0

IN_NODE, HID_NODE = None, None
hid, out = None, None
delta_hid, delta_out = None, None

epoch, days = 0, 0
fError = 0.05

x, t = None, None
v, w = [], []

isPlot = True


def findHidOut(n: int):
    for i in range(HID_NODE):
        dot_h = 0
        for j in range(IN_NODE):
            dot_h += x[n][j] * v[i][j]
        hid[i] = sigmoid(dot_h)

    hid[HID_NODE - 1] = 0.5  # random.random()

    for i in range(OUT_NODE):
        dot_o = 0
        for j in range(HID_NODE):
            dot_o += w[i][j] * hid[j]
        out[i] = sigmoid(dot_o)


def printResult(DIV_T: float):
    arrNet = []
    s = 0
    s_max = -256
    s_min = 256
    for i in range(days):
        findHidOut(i)

        undo_out = round(out[0] * DIV_T, 2)
        undo_teacher = round(t[i][0] * DIV_T, 2)

        pad_out = str(undo_out).rjust(6)
        pad_teacher = str(undo_teacher).rjust(6)

        net = 100 * (t[i][0] - out[0]) / t[i][0]
        s += net
        arrNet.append(net)
        str_date = arrHsh[i]["date"]
        print(f"{str_date} {pad_out} True: {pad_teacher} valance: {round(s, 2)}")

        if s_max < s:
            s_max = s
        if s < s_min:
            s_min = s

    rd_err = round(fError, 5)
    s_max = round(s_max, 2)
    s_min = round(s_min, 2)
    s_mean = (s_max + s_min) / 2
    s_mean = round(s_mean, 2)
    f_time = time_ed - time_st

    if 60 <= f_time:
        n_minute = int(f_time / 60)
        f_sec = round(f_time % 60)
    else:
        n_minute = 0
        f_sec = round(f_time, 2)

    print(f"epoch: {epoch} final err: {rd_err} days: {days}")
    print(f"max: {s_max} min: {s_min} mean: {s_mean}")
    print(f"time: {n_minute} min {f_sec} sec.")


def addBias(hsh: dict) -> dict:
    arrInput = hsh["input"]
    arrInput.append(-1)  # add bias
    return arrInput


if __name__ == "__main__":
    f = open("./json/n225out.json", "r")  # xor | cell30 | py225 | n225out
    arrHsh = json.load(f)
    f2 = open("./json/setting.json", "r")
    HshSetting = json.load(f2)

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
            v[i].append(0.5)  # random() | uniform(0.5, 1.0)
    for i in range(OUT_NODE):
        for j in range(HID_NODE):
            w[i].append(0.5)  # random() | uniform(0.5, 1.0)

    date_now = datetime.datetime.now()
    print(date_now.strftime("%F %T"))

    time_st = time.time()

    while epoch < THRESHOLD:
        epoch += 1
        fError = 0

        for n in range(days):
            findHidOut(n)

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
    printResult(HshSetting["DIV_T"])
    # show plot
    if isPlot:
        plt.plot(arrErr)
        plt.show()