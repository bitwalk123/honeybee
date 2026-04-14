import os
from os.path import expanduser

from matplotlib import dates as mdates
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    df = pd.read_pickle("performance.pkl")
    total = df["pnl"].sum()
    days = len(df)
    print(df)

    list_code = list(set(df["code"]))
    if len(list_code) == 1:
        code = list_code[0]
    else:
        raise ValueError(f"len(list_code) is not 1!")

    # Matplotlib の共通設定
    FONT_PATH = "fonts/RictyDiminished-Regular.ttf"
    fm.fontManager.addfont(FONT_PATH)

    # FontPropertiesオブジェクト生成（名前の取得のため）
    font_prop = fm.FontProperties(fname=FONT_PATH)
    font_prop.get_name()

    plt.rcParams["font.family"] = font_prop.get_name()

    fig = plt.figure(figsize=(6.8, 4))
    ax = dict()
    n = 2
    gs = fig.add_gridspec(
        n, 1, wspace=0.0, hspace=0.0,
        height_ratios=[2 if i == 0 else 1 for i in range(n)]
    )
    for i, axis in enumerate(gs.subplots(sharex="col")):
        ax[i] = axis
        ax[i].grid()

    ax[0].set_title(f"推論パフォーマンス ({code}) : 総収益 {total} 円/株 in {days} 日")
    ax[0].plot(df["pnl"], color="C0")
    ax[0].set_ylabel("損益/株")
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax[0].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

    ax[1].plot(df["contracts"], color="C1")
    ax[1].set_ylabel("約定回数")

    plt.tight_layout()
    output = "performance.png"
    plt.savefig(output)
    plt.show()
