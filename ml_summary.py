import os
import itertools as ittl
import numpy as np
import pandas as pd
from skyrim.falkreath import CManagerLibReader, CTable
from skyrim.riften import CNAV


def cal_precision_and_recall(t_value: int, t_y_actu: np.ndarray, t_y_pred: np.ndarray):
    _obs = len(t_y_actu)
    _tp = sum(t_y_actu[t_y_pred == t_value] == t_value)
    _fp = sum(t_y_actu[t_y_pred == t_value] != t_value)
    _fn = sum(t_y_actu[t_y_pred != t_value] == t_value)
    _tn = sum(t_y_actu[t_y_pred != t_value] != t_value)

    _p = (_tp + _fn) / _obs
    pos_prec = _tp / (_tp + _fp) if _tp + _fp > 0 else 0
    pos_reca = _tp / (_tp + _fn) if _tp + _fn > 0 else 0

    _q = (_tn + _fp) / _obs
    neg_prec = _tn / (_tn + _fn) if _tn + _fn > 0 else 0
    neg_reca = _tn / (_tn + _fp) if _tn + _fp > 0 else 0

    win_rate = (_tp + _tn) / _obs

    return {
        "obs": _obs, "win_rate": win_rate,
        "p": _p, "pos_precision": pos_prec, "pos_recall": pos_reca,
        "q": _q, "neg_precision": neg_prec, "neg_recall": neg_reca,
    }


def cal_trades(t_raw_df: pd.DataFrame, t_cost_rate: float):
    ret_df = t_raw_df[["trade_date", "raw_ret"]].groupby("trade_date")[["raw_ret"]].apply(lambda z: z.mean(axis=0))
    ret_df["net_ret"] = ret_df["raw_ret"] - t_cost_rate
    ret_df["nav"] = (ret_df["net_ret"] + 1).cumprod()
    nav = CNAV(t_raw_nav_srs=ret_df["net_ret"], t_annual_rf_rate=0, t_type="RET", t_freq="D")
    nav.cal_all_indicators()
    return nav.to_dict(t_type="eng"), ret_df[["net_ret", "nav"]]


def ml_summary_model(model_lbl: str,
                     instrument: str | None, tid: str, trn_win: int,
                     predictions_dir: str, navs_dir: str,
                     sqlite3_tables: dict,
                     cost_rate: float, ret_scale: int = 100
                     ):
    model_grp_id = "-".join(filter(lambda z: z, ["M", instrument, tid, "TMW{:02d}".format(trn_win)]))
    pred_id = model_grp_id + "-pred-{}".format(model_lbl)
    predictions_lib = CManagerLibReader(
        t_db_save_dir=predictions_dir,
        t_db_name=pred_id + ".db",
    )
    predictions_lib_stru = sqlite3_tables[pred_id]
    predictions_lib_tab = CTable(t_table_struct=predictions_lib_stru)
    predictions_lib.set_default(predictions_lib_tab.m_table_name)
    predictions_df = predictions_lib.read(
        t_value_columns=["trade_date", "instrument", "contract", "tid", "timestamp", "rtm", "pred"]
    )

    if model_lbl in ["mlpc"]:
        predictions_df["pred"] = predictions_df["pred"] * 2 - 1

    summary_header = {
        "model": model_lbl,
        "instrument": instrument,
        "tid": tid,
        "tmw": trn_win,
    }

    # classify models
    cls_df = predictions_df[["rtm", "pred"]].applymap(lambda z: 1 if z >= 0 else 0)
    summary_model = summary_header.copy()
    summary_model.update(
        cal_precision_and_recall(t_value=1, t_y_actu=cls_df["rtm"], t_y_pred=cls_df["pred"]))

    # simu trades
    predictions_df["raw_ret"] = np.sign(predictions_df["pred"]) * predictions_df["rtm"] / ret_scale
    summary_trades = summary_header.copy()
    sum_trades, nav_df = cal_trades(t_raw_df=predictions_df, t_cost_rate=cost_rate)
    summary_trades.update(sum_trades)
    nav_file = "{}-nav.csv.gz".format(pred_id)
    nav_path = os.path.join(navs_dir, nav_file)
    nav_df.to_csv(nav_path, float_format="%.8f")
    return summary_model, summary_trades


def ml_summary(model_lbl: str,
               instruments_universe: list[str], tids: list[str], train_windows: list[int],
               sqlite3_tables: dict,
               predictions_dir: str, navs_dir: str,
               research_summary_dir: str,
               cost_rate: float,
               ):
    res_models, res_trades = [], []
    for instrument, tid, train_window in ittl.product(instruments_universe + [None], tids, train_windows):
        ans_model, ans_trade = ml_summary_model(
            model_lbl=model_lbl,
            instrument=instrument, tid=tid, trn_win=train_window,
            predictions_dir=predictions_dir, navs_dir=navs_dir,
            sqlite3_tables=sqlite3_tables,
            cost_rate=cost_rate
        )
        res_models.append(ans_model)
        res_trades.append(ans_trade)
        print("... | {:>8s} | {:>8s} | {:>3s} | TMW{:02d} | summarized |".format(
            model_lbl,
            instrument if instrument else "",
            tid if tid else "",
            train_window))
    res_models_df, res_trades_df = pd.DataFrame(res_models), pd.DataFrame(res_trades)
    res_trades_df["sharpe_ratio"] = res_trades_df["sharpe_ratio"].astype(float)

    res_models_file = "summary.{}.models.csv".format(model_lbl)
    res_trades_file = "summary.{}.trades.csv".format(model_lbl)
    res_models_path = os.path.join(research_summary_dir, res_models_file)
    res_trades_path = os.path.join(research_summary_dir, res_trades_file)
    res_models_df.to_csv(
        res_models_path, index=False, float_format="%.6f")
    res_trades_df.sort_values(by="sharpe_ratio", ascending=False).head(20).to_csv(
        res_trades_path, index=False, float_format="%.2f")
    return 0
