# 疑似モデルを利用した推論（単一ファイル）
import glob
import os

from tools.gnat import Gnat

if __name__ == "__main__":

    obj = Gnat({})

    # 学習に使用するティックデータ
    home = os.path.expanduser("~")
    path_excel = os.path.join(home, "MyProjects", "kabuto", "collection", "*.xlsx")
    list_file_excel = sorted(glob.glob(path_excel))
    for file_excel in list_file_excel[:5]:
        print("推論対象ファイル")
        print(file_excel)
        obj.run(file_excel)
        #obj.plot()
        obj.show_transaction()