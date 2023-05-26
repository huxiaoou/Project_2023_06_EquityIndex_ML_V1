import itertools as ittl

equity_indexes = (
    ("000016.SH", "IH.CFE"),
    ("000300.SH", "IF.CFE"),
    ("000905.SH", "IC.CFE"),
    ("000852.SH", "IM.CFE"),
)

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
        "value_columns": {
            "basis": "REAL",
            "csr": "REAL",
            "onr": "REAL",
            "vwap_ret": "REAL",
            "vwap_cum_ret": "REAL",
            "hgh_ret": "REAL",
            "low_ret": "REAL",
            "vtop01_ret": "REAL",
            "vtop02_ret": "REAL",
            "vtop05_ret": "REAL",
            "vtop01_cvp": "REAL",
            "vtop02_cvp": "REAL",
            "vtop05_cvp": "REAL",
            "vtop01_cvr": "REAL",
            "vtop02_cvr": "REAL",
            "vtop05_cvr": "REAL",
            "cvp": "REAL",
            "cvr": "REAL",
            "up": "REAL",
            "dn": "REAL",
            "skewness": "REAL",
            "smart01": "REAL",
            "smart02": "REAL",
            "smart05": "REAL",
            "smart01_ret": "REAL",
            "smart02_ret": "REAL",
            "smart05_ret": "REAL",

            "rtm": "REAL",
        }
    },

}

x_lbls = ["alpha{:02d}".format(_) for _ in range(19)]
y_lbls = ["rtm"]
instruments_universe = ["IC.CFE", "IH.CFE", "IF.CFE", "IM.CFE"]
tids = ["T{:02d}".format(t) for t in range(1, 8)]

# train_windows = (6, 12, 24)
# model_lbls = ["lm", "mlpc", "mlpr"]
# for instrument, tid, trn_win, model_lbl in ittl.product(
#         instruments_universe + [None], tids, train_windows, model_lbls):
#     model_grp_id = "-".join(filter(lambda z: z, ["M", instrument, tid, "TMW{:02d}".format(trn_win)]))
#     pred_id = model_grp_id + "-pred-{}".format(model_lbl)
#     sqlite3_tables.update({
#         pred_id: {
#             "table_name": "predictions",
#             "primary_keys": {
#                 "trade_date": "TEXT",
#                 "instrument": "TEXT",
#                 "contract": "TEXT",
#                 "tid": "TEXT",
#                 "timestamp": "INT4",
#             },
#             "value_columns": {
#                 "rtm": "REAL",
#                 "pred": "REAL",
#             }
#         },
#     })
cost_rate = 5e-4
