import pandas as pd
from matplotlib import (
    dates as mdates,
    font_manager as fm,
    pyplot as plt,
    ticker as ticker,
)


def learning_curve(df: pd.DataFrame, subtitle: str):
    # 報酬のプロット
    fig, ax = plt.subplots(figsize=(6.8, 4))

    ax.plot(df["r"])
    ax.set_title(f"Learning Curve\n{subtitle}")
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.grid()
    plt.tight_layout()
    plt.savefig("trend_reward.png")
    #plt.show()
    plt.close()


def plot_performance(code: str, df: pd.DataFrame):
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

    total = df["pnl"].sum()
    days = len(df)
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
