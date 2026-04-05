import datetime
import re

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
