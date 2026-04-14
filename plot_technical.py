import pandas as pd
from matplotlib import (
    dates as mdates,
    font_manager as fm,
    pyplot as plt,
    ticker as ticker,
)

if __name__ == "__main__":
    df = pd.read_pickle("technical.pkl")
    print(df.columns)
    dt = df.index[0]
    print(dt.date())

    # Matplotlib の共通設定
    FONT_PATH = "fonts/RictyDiminished-Regular.ttf"
    fm.fontManager.addfont(FONT_PATH)

    # FontPropertiesオブジェクト生成（名前の取得のため）
    font_prop = fm.FontProperties(fname=FONT_PATH)
    font_prop.get_name()

    plt.rcParams["font.family"] = font_prop.get_name()

    fig = plt.figure(figsize=(6.8, 4))
    ax = dict()
    n = 3
    gs = fig.add_gridspec(
        n, 1, wspace=0.0, hspace=0.0,
        height_ratios=[2 if i == 0 else 1 for i in range(n)]
    )
    for i, axis in enumerate(gs.subplots(sharex="col")):
        ax[i] = axis
        ax[i].grid()

    ax[0].plot(df["price"],linewidth=1, label="株価")
    ax[0].plot(df["ma1"],linewidth=1, label="MA1")
    ax[0].set_ylabel("株価")
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax[0].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax[0].legend(fontsize=6)

    ax[1].plot(df["diff_vwap"],linewidth=1)

    ax[2].plot(df["profit"],linewidth=1)

    plt.tight_layout()
    output = "technical.png"
    plt.savefig(output)
    plt.show()
