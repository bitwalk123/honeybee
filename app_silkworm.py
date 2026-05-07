# DOE 結果の解析
import os

import pandas as pd

from tools.silkworm import SilkWorm

if __name__ == "__main__":
    name_doe = "doe-003"

    csv_result = os.path.join("doe", name_doe, "result.csv")  # 結果用
    # 解析対象ファイル
    print("解析対象ファイル :", csv_result)
    df = pd.read_csv(csv_result)
    obj = SilkWorm(name_doe, df)
    # obj.mulreg()
    #obj.main_effect()
    obj.ranking()
