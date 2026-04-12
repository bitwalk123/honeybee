import os

import pandas as pd


def get_excel_sheet(path_excel: str, sheet: str) -> pd.DataFrame:
    """
    指定したExcelファイルの指定したシートをデータフレームに読み込む
    :param path_excel:
    :param sheet:
    :return:
    """
    if os.path.isfile(path_excel):
        wb = pd.ExcelFile(path_excel)
        list_sheet = wb.sheet_names
        if sheet in list_sheet:
            return wb.parse(sheet_name=sheet)
        else:
            return pd.DataFrame()
    else:
        return pd.DataFrame()
