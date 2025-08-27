import pandas as pd

def compute_cprs(aprs: pd.Series, sgprs: pd.Series, cbprs: pd.Series, w_aprs: float, w_sgprs: float, w_cbprs: float) -> pd.DataFrame:
    df = pd.concat([aprs, sgprs, cbprs], axis=1).fillna(0.0)
    df.columns = ["APRS", "SGPRS", "CBPRS"]
    df["CPRS"] = w_aprs * df["APRS"] + w_sgprs * df["SGPRS"] + w_cbprs * df["CBPRS"]
    return df
