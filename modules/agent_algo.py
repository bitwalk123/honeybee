from collections import defaultdict

from funcs.excel import get_excel_sheet
from modules.env_inference import InferenceEnv
from modules.model_algo import AlgoModel


class AlgoAgent:
    def __init__(self, code: str, ) -> None:
        self.code: str = code

    def infer(self, file_excel: str) -> tuple:
        # 指定銘柄コードのティックデータのデータフレームを取得
        df = get_excel_sheet(file_excel, self.code)

        # 1. 環境クラス継承の推論用環境クラスのインスタンス
        env = InferenceEnv(self.code, df)

        # 2. アルゴリズム・モデル
        model = AlgoModel()

        # ====== 推論実施 ======
        print("Begin inference...")
        dict_result = dict()
        dict_technical = defaultdict(list)

        # 環境のリセット
        obs, _ = env.reset()
        # print(obs)
        episode_over = False

        info = []
        while not episode_over:
            # マスク情報付きで推論
            action_masks = env.action_masks()
            action, _states = model.predict(obs, action_masks=action_masks)
            # 環境でステップ処理
            obs, reward, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated
            if "technical" in info:
                d = info["technical"]
                for key in d.keys():
                    dict_technical[key].append(d[key])
        else:
            # 取引結果を出力
            if "transaction" in info:
                dict_result["transaction"] = info["transaction"]

        # 環境の終了処理
        env.close()
        return dict_result, dict_technical
