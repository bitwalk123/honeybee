# 疑似モデルを利用した推論（単一ファイル）
import glob
import os
import re

from tools.gnat import Gnat

if __name__ == "__main__":
    obj = Gnat({})

    # 学習に使用するティックデータ
    home = os.path.expanduser("~")
    path_excel = os.path.join(home, "MyProjects", "kabuto", "collection", "*.xlsx")
    list_file_excel = sorted(glob.glob(path_excel))
    file_excel = list_file_excel[-1]
    print("推論対象ファイル")
    print(file_excel)
    obj.run(file_excel)
    if m := re.search(r"(\d{4})(\d{2})(\d{2})", file_excel):
        path_date = str(os.path.join(*m.groups()))
    else:
        path_date = "0000/00/00"
    obj.plot(path_date)
    obj.show_transaction()
