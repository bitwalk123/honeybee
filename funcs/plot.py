import pandas as pd
from matplotlib import pyplot as plt


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
