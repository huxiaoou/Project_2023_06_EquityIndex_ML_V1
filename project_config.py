import itertools as ittl

equity_indexes = (
    ("000016.SH", "IH.CFE"),
    ("000300.SH", "IF.CFE"),
    ("000905.SH", "IC.CFE"),
    ("000852.SH", "IM.CFE"),
)

factors = [
    "basis",
    "csr",
    "onr",
    "vwap_ret",
    "vwap_cum_ret",
    "hgh_ret",
    "low_ret",
    "vtop01_ret",
    "vtop02_ret",
    "vtop05_ret",
    "vtop01_cvp",
    "vtop02_cvp",
    "vtop05_cvp",
    "vtop01_cvr",
    "vtop02_cvr",
    "vtop05_cvr",
    "cvp",
    "cvr",
    "up",
    "dn",
    "skewness",
    "smart01",
    "smart01_ret",
    "smart02",
    "smart02_ret",
    "smart05",
    "smart05_ret",
    "vh01",
    "vl01",
    "vd01",
    "vh02",
    "vl02",
    "vd02",
    "vh05",
    "vl05",
    "vd05",
    "exr",
    "exrb01",
    "gu",
    "gd",
    "g_tau",
    "g_tau_abs",
    "mtm_vol_adj",
]

sqlite3_tables = {
    "features_and_return": {
        "table_name": "features_and_return",
        "primary_keys": {
            "trade_date": "TEXT",
            "instrument": "TEXT",
            "contract": "TEXT",
            "tid": "TEXT",
            "timestamp": "INT4",
        },
        "value_columns": {f: "REAL" for f in factors + ["rtm"]}
    },

}

instruments_universe = ["IC.CFE", "IH.CFE", "IF.CFE", "IM.CFE"]
tids = ["T{:02d}".format(t) for t in range(1, 8)]

# --- ic tests
for factor, tid in ittl.product(factors, tids):
    ic_tests_lib_id = "{}-{}-ic_tests".format(factor, tid)
    sqlite3_tables.update({
        ic_tests_lib_id: {
            "table_name": "ic_tests",
            "primary_keys": {
                "trade_date": "TEXT",
            },
            "value_columns": {
                "obs": "INTEGER",
                "ic": "REAL",
            }
        },
    })

# --- ic tests
for factor, tid in ittl.product(factors, tids):
    group_tests_lib_id = "{}-{}-group_tests".format(factor, tid)
    sqlite3_tables.update({
        group_tests_lib_id: {
            "table_name": "ic_tests",
            "primary_keys": {
                "trade_date": "TEXT",
            },
            "value_columns": {
                "lng": "REAL",
                "srt": "REAL",
                "hdg": "REAL",
            }
        },
    })

# --- simulation
cost_rate = 5e-4

if __name__ == "__main__":
    print("Number of factors = {}".format(len(factors)))
