import pandas as pd
from matplotlib import (
    font_manager as fm,
    pyplot as plt,
)

from funcs.plot import (
    plot_diff_ma,
    plot_diff_vwap,
    plot_main,
    plot_profit,
)
from funcs.tide import get_tse_x_range

if __name__ == "__main__":
    code = "9984"
    df = pd.read_pickle("technical.pkl")

    # プロットの x軸の範囲を算出（左右10分のマージン）
    dt_date, dt_left, dt_right = get_tse_x_range(df)

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
    title = f"{dt_date} : {code} の推論パフォーマンス, 損益 {pnl} 円/株"
    ax[i].set_xlim(dt_left, dt_right)
    plot_main(ax[i], df, title)

    # MA乖離率
    i += 1
    plot_diff_ma(ax[i], df)

    # VWAP乖離率
    i += 1
    plot_diff_vwap(ax[i], df)

    # 含み損益
    i += 1
    plot_profit(ax[i], df)

    plt.tight_layout()
    output = "technical.png"
    plt.savefig(output)
    plt.show()
