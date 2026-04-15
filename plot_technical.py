import pandas as pd
from matplotlib import (
    dates as mdates,
    font_manager as fm,
    pyplot as plt,
    ticker as ticker,
)

if __name__ == "__main__":
    code = "9984"
    df = pd.read_pickle("technical.pkl")
    print(df.columns)
    dt = df.index[0]
    print(dt.date())

    df_transaction = pd.read_pickle("transaction.pkl")
    r = df_transaction.index[-1]
    pnl = df_transaction.tail(1).loc[r, "pnl"]

    # Matplotlib の共通設定
    FONT_PATH = "fonts/RictyDiminished-Regular.ttf"
    fm.fontManager.addfont(FONT_PATH)

    # FontPropertiesオブジェクト生成（名前の取得のため）
    font_prop = fm.FontProperties(fname=FONT_PATH)
    font_prop.get_name()

    plt.rcParams["font.family"] = font_prop.get_name()

    fig = plt.figure(figsize=(6.8, 6))
    ax = dict()
    n = 4
    gs = fig.add_gridspec(
        n, 1,
        wspace=0.0, hspace=0.0,
        height_ratios=[2 if i == 0 else 1 for i in range(n)]
    )
    for i, axis in enumerate(gs.subplots(sharex="col")):
        ax[i] = axis
        ax[i].grid()

    # 株価
    i = 0
    ax[i].set_title(f"{dt.date()} : {code} の推論パフォーマンス, 損益 {pnl} 円/株")
    ax[i].plot(df["price"], color="black", alpha=0.25, linewidth=0.5, zorder=10, label="株価")
    ax[i].plot(df["ma1"], linewidth=1, zorder=20, label="MA1")
    ax[i].plot(df["ma2"], linewidth=1, zorder=30, label="MA2")
    ax[i].plot(df["vwap"], linewidth=0.75, zorder=40, label="VWAP")
    ax[i].set_ylabel("株価")
    ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax[i].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax[i].legend(fontsize=6)

    # MA乖離率
    i += 1
    ax[i].plot(df["diff_ma"], color="black", alpha=0.5, linewidth=0.75)
    ax[i].axhline(y=0, color="black", linewidth=0.75, alpha=1, zorder=0)
    ax[i].set_ylabel("MA乖離率")

    # VWAP乖離率
    i += 1
    ax[i].plot(df["diff_vwap"], color="black", alpha=0.5, linewidth=0.75)
    ax[i].axhline(y=0, color="black", linewidth=0.75, alpha=1, zorder=0)
    ax[i].set_ylabel("VWAP乖離率")

    # 含み損益
    i += 1
    x = df.index
    y1 = df["profit"]
    ax[i].plot(df["profit"], linewidth=0.5, color="black", alpha=0.1, zorder=10)
    ax[i].fill_between(x, 0, y1, where=(0 < y1), fc="#fbb", ec="#f00", alpha=0.5, lw=0.5, zorder=20, label="含み益")
    ax[i].fill_between(x, 0, y1, where=(y1 < 0), fc="#bbf", ec="#00f", alpha=0.5, lw=0.5, zorder=20, label="含み損")
    ax[i].set_ylabel("含み損益")

    plt.tight_layout()
    output = "technical.png"
    plt.savefig(output)
    plt.show()
