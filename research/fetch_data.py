"""Fetch CFTC legacy futures-only COT and daily prices for the playbook study.

Data are cached outside the repo (public sources, re-fetchable).
CFTC: https://publicreporting.cftc.gov/resource/6dca-aqww.json (legacy, futures only)
Prices: Yahoo Finance continuous front-month via yfinance (NOT back-adjusted).
"""
import json, os, sys, time
import pandas as pd
import requests

OUT = os.environ.get("BHWS_RESEARCH_DATA", "/home/user/workspace/bhws_upgrade/research_data")
MARKETS = {
    # id: (cftc code, yahoo ticker, group)
    "6E": ("099741", "6E=F", "fx"), "6B": ("096742", "6B=F", "fx"), "6J": ("097741", "6J=F", "fx"),
    "6A": ("232741", "6A=F", "fx"), "6C": ("090741", "6C=F", "fx"), "6S": ("092741", "6S=F", "fx"),
    "6N": ("112741", "6N=F", "fx"), "DX": ("098662", "DX-Y.NYB", "fx"),
    "ES": ("13874A", "ES=F", "equity"), "NQ": ("209742", "NQ=F", "equity"),
    "ZB": ("020601", "ZB=F", "rates"), "ZN": ("043602", "ZN=F", "rates"),
    "GC": ("088691", "GC=F", "metals"), "SI": ("084691", "SI=F", "metals"), "HG": ("085692", "HG=F", "metals"),
    "CL": ("067651", "CL=F", "energy"), "NG": ("023651", "NG=F", "energy"),
    "ZC": ("002602", "ZC=F", "ags"), "ZS": ("005602", "ZS=F", "ags"), "ZW": ("001602", "ZW=F", "ags"),
    "SB": ("080732", "SB=F", "softs"), "KC": ("083731", "KC=F", "softs"), "CT": ("033661", "CT=F", "softs"),
    "CC": ("073732", "CC=F", "softs"), "LE": ("057642", "LE=F", "meats"), "HE": ("054642", "HE=F", "meats"),
}


BASIS = os.environ.get("BHWS_COT_BASIS", "futures_only")   # or "combined"
DATASETS = {"futures_only": "6dca-aqww", "combined": "jun7-fc8e"}
PREFIX = "cot_" if BASIS == "futures_only" else "cotcomb_"


def cot(code):
    url = f"https://publicreporting.cftc.gov/resource/{DATASETS[BASIS]}.json"
    q = {"$select": "report_date_as_yyyy_mm_dd,noncomm_positions_long_all,noncomm_positions_short_all,"
                    "comm_positions_long_all,comm_positions_short_all,open_interest_all",
         "$where": f"cftc_contract_market_code='{code}' AND report_date_as_yyyy_mm_dd >= '2003-01-01'",
         "$order": "report_date_as_yyyy_mm_dd ASC", "$limit": 5000}
    r = requests.get(url, params=q, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"]).dt.date
    for c in df.columns:
        if c.endswith("_all"):
            df[c] = pd.to_numeric(df[c])
    df["mgr_net"] = df.noncomm_positions_long_all - df.noncomm_positions_short_all
    df["comm_net"] = df.comm_positions_long_all - df.comm_positions_short_all
    return df[["date", "mgr_net", "comm_net", "open_interest_all"]].drop_duplicates("date")


def main():
    import yfinance as yf
    log = {}
    for mid, (code, yt, grp) in MARKETS.items():
        try:
            c = cot(code)
            c.to_csv(f"{OUT}/{PREFIX}{mid}.csv", index=False)
            log[mid] = {"cot_rows": len(c), "cot_last": str(c.date.iloc[-1])}
            if BASIS != "futures_only":      # prices already cached by the futures-only run
                print(mid, log[mid], flush=True); time.sleep(1.5); continue
            p = yf.download(yt, start="2003-01-01", interval="1d", progress=False, auto_adjust=False)
            if isinstance(p.columns, pd.MultiIndex):
                p.columns = p.columns.get_level_values(0)
            p = p[["Close"]].dropna()
            p.to_csv(f"{OUT}/px_{mid}.csv")
            log[mid] = {"cot_rows": len(c), "cot_first": str(c.date.iloc[0]), "cot_last": str(c.date.iloc[-1]),
                        "px_rows": len(p), "px_first": str(p.index[0].date()), "px_last": str(p.index[-1].date())}
        except Exception as e:
            log[mid] = {"error": repr(e)[:200]}
        print(mid, log[mid], flush=True)
        time.sleep(1.5)
    json.dump(log, open(f"{OUT}/fetch_log_{BASIS}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
