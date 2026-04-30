from collections import defaultdict

from funcs.excel import get_excel_sheet
from modules.env_inference import InferenceEnv


class AlgoModel:
    def __init__(self):
        pass


class AlgoAgent:
    def __init__(self, code: str, ) -> None:
        self.code: str = code

    def infer(self, file_excel: str) -> tuple:
        # 指定銘柄コードのティックデータのデータフレームを取得
        self.df = get_excel_sheet(file_excel, self.code)

        # 1. 環境クラス継承の推論用環境クラスのインスタンス
        env = InferenceEnv(self.code, self.df)

        # 2. アルゴリズム・モデル
        model = AlgoModel()

        # ====== 推論実施 ======
        print("Begin inference...")
        dict_result = dict()
        dict_technical = defaultdict(list)

        # 環境のリセット
        obs = env.reset()
        episode_over = False

        info = []
        while not episode_over:
            action_masks = env.action_masks()  # バッチ次元を付与
            # マスク情報付きで推論
            action, _states = model.predict(
                obs,
                action_masks=action_masks,
                deterministic=False
            )
            # 環境でステップ処理
            # action = np.array([action])  # VecEnv では複数環境分の配列
            obs, reward, done, info = env.step(action)
            # print(obs, reward, done, info)
            # total_reward += reward[idx]
            episode_over = done[idx]
            if "technical" in info[idx]:
                d = info[idx]["technical"]
                for key in d.keys():
                    dict_technical[key].append(d[key])
        else:
            dict_info = info[idx]
            # 取引結果を出力
            if "transaction" in dict_info:
                dict_result["transaction"] = dict_info["transaction"]

        # 環境の終了処理
        env.close()
        return dict_result, dict_technical
