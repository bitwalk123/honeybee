import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def simulate_pursuit(data_series, look_ahead, alpha=0.2):
    """
    Pure Pursuit的概念を用いた追従シミュレーション
    data_series: ティックデータの時系列（pd.Series）
    look_ahead: 前方注視距離（何ステップ先を目標にするか）
    alpha: 追従ゲイン（どれくらい目標に近づくか。0.1〜0.5程度で調整）
    """
    data = data_series.values
    path = np.zeros_like(data)
    current_pos = data[0]  # 初期位置

    for i in range(len(data)):
        # 1. 前方注視距離(Look-ahead)に基づきターゲットを決定
        target_idx = min(i + look_ahead, len(data) - 1)
        target_val = data[target_idx]

        # 2. 現在位置からターゲットに向かって移動（追従制御）
        # ロボットが目標点に向かってハンドルを切る動作に相当
        current_pos = current_pos + alpha * (target_val - current_pos)
        path[i] = current_pos

    return path


# --- サンプルデータの作成 (お手持ちのデータがある場合は読み替えてください) ---
# df = pd.read_csv('your_tick_data.csv')
# tick_prices = df['price']

# ここではデモ用にサイン波+ノイズで生成
np.random.seed(42)
steps = 200
raw_ticks = 100 + np.sin(np.linspace(0, 10, steps)) * 5 + np.random.normal(0, 0.5, steps)
tick_series = pd.Series(raw_ticks)

# --- 追従線の計算 ---
# 短い先読み (感度重視: Short-term Look-ahead)
path_sensitive = simulate_pursuit(tick_series, look_ahead=3, alpha=0.3)

# 長い先読み (滑らかさ重視: Long-term Look-ahead)
path_smooth = simulate_pursuit(tick_series, look_ahead=15, alpha=0.1)

# --- プロット ---
plt.figure(figsize=(12, 6))
plt.plot(tick_series, label='Original Tick Data', color='silver', alpha=0.6)
plt.plot(path_sensitive, label='Short Look-ahead (Look=3)', color='crimson', lw=2)
plt.plot(path_smooth, label='Long Look-ahead (Look=15)', color='royalblue', lw=2)

plt.title('Tick Data Tracking with Pure Pursuit Logic')
plt.xlabel('Tick Step')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
