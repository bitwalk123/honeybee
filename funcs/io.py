import datetime
import os
import re
import shutil

import pandas as pd
from pandas.io.parsers import TextFileReader


def get_sample_data(file_csv: str) -> tuple[str, pd.DataFrame]:
    pattern = re.compile(r"[0-9]{8}_([A-Z0-9]{4})\.csv")
    if m := pattern.match(file_csv):
        code: str = m.group(1)
    else:
        code: str = "0000"
    df = pd.read_csv(file_csv)
    df.index = [datetime.datetime.fromtimestamp(t) for t in df["Time"]]
    df.index.name = "Datetime"
    return code, df


def update_new_dir(name_dir: str):
    """
    新たにディレクトリを作成し直す
    :param name_dir:
    :return:
    """
    if os.path.exists(name_dir):
        shutil.rmtree(name_dir)
    os.makedirs(name_dir, exist_ok=True)


def prep_dir_logs_monitor(name_dir: str) -> str:
    update_new_dir(name_dir)
    file_log = os.path.join(name_dir, "monitor.csv")
    return file_log
