import datetime
import re


def get_dt_from_excel(excel_path) -> datetime.datetime:
    pattern = re.compile(r".+_([0-9]{4})([0-9]{2})([0-9]{2}).*\.xlsx")
    if m := pattern.match(excel_path):
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))
    else:
        year = 1970
        month = 1
        day = 1
    return datetime.datetime(year, month, day)


def get_tse_x_range(df):
    dt = df.index[0]
    dt_date = dt.date()
    t_left = datetime.time(8, 50)
    t_right = datetime.time(15, 40)
    dt_left = datetime.datetime.combine(dt_date, t_left)
    dt_right = datetime.datetime.combine(dt_date, t_right)
    return dt_date, dt_left, dt_right
