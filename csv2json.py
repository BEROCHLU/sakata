#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd


def run():
    folder_path = "T:\\ProgramFilesT\\pleiades\\workspace\\node225"
    csv_path = os.path.join(folder_path, "nt1570.csv")

    df = pd.read_csv(csv_path)
    df.to_json("./json/nt1570.json", orient="records")


if __name__ == "__main__":
    run()
