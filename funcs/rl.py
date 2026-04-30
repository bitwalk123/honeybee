import os

def get_transaction(code:str, f: str, r: dict, d: dict) -> None:
    df = r["transaction"]
    print(df)
    filename = os.path.basename(f)
    pnl = df["損益"].sum()
    n_contract = len(df)
    print(f"{filename}, 損益 : {pnl} 円, 約定回数 : {n_contract} 回")

    d["file"].append(filename)
    d["code"].append(code)
    d["pnl"].append(pnl)
    d["contracts"].append(n_contract)
