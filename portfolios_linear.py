import os
import datetime as dt
import numpy as np
import pandas as pd
import multiprocessing as mp
from skyrim.falkreath import CManagerLibReader, CTable
from skyrim.winterhold import plot_lines


def cal_portfolio_return(df: pd.DataFrame, selected_factors_df: pd.DataFrame, ret: str, ret_scale: int):
    n = len(df)
    m = int(n / 2)
    d = n - 2 * m
    lng_raw_wgt = np.array([1] * m + [0] * d + [0] * m)
    srt_raw_wgt = np.array([0] * m + [0] * d + [1] * m)
    lng_wgt = lng_raw_wgt / np.abs(lng_raw_wgt).sum()
    srt_wgt = srt_raw_wgt / np.abs(srt_raw_wgt).sum()
    hdg_wgt = lng_wgt * 0.5 - srt_wgt * 0.5
    raw_wgt_data = {}
    for fac, sig in selected_factors_df.itertuples(index=False):
        sig_df = df[["instrument", fac]].sort_values(by=fac, ascending=False if sig > 0 else True)
        raw_wgt_data[fac] = pd.Series(hdg_wgt, index=sig_df["instrument"])
    raw_wgt_df = pd.DataFrame(raw_wgt_data)
    tot_wgt_srs = raw_wgt_df.sum(axis=1)
    tot_wgt_srs_abs_sum = tot_wgt_srs.abs().sum()
    wgt_srs = tot_wgt_srs / tot_wgt_srs_abs_sum
    port_ret = df[["instrument", ret]].set_index("instrument")[ret] @ wgt_srs / ret_scale
    return port_ret


def portfolios_linear_per_tid(
        tid: str,
        bgn_date: str, stp_date: str | None,
        features_and_return_dir: str,
        group_tests_summary_dir: str,
        sqlite3_tables: dict,
):
    # --- load lib reader
    features_and_return_lib = CManagerLibReader(
        t_db_save_dir=features_and_return_dir,
        t_db_name="features_and_return.db"
    )
    features_and_return_db_stru = sqlite3_tables["features_and_return"]
    features_and_return_tab = CTable(t_table_struct=features_and_return_db_stru)
    features_and_return_lib.set_default(features_and_return_tab.m_table_name)

    selected_factors_df = pd.read_csv(
        os.path.join(group_tests_summary_dir, "selected-factors-{}.csv.gz".format(tid))
    )
    selected_factors = selected_factors_df["factor"].tolist()

    src_df = features_and_return_lib.read_by_conditions(
        t_conditions=[
            ("trade_date", ">=", bgn_date),
            ("trade_date", "<", stp_date),
            ("tid", "=", tid),
        ],
        t_value_columns=["trade_date", "instrument", "tid"] + selected_factors + ["rtm"]
    )

    portfolio_ret_srs = src_df.groupby(by="trade_date").apply(
        cal_portfolio_return, selected_factors_df=selected_factors_df, ret="rtm", ret_scale=100)

    return portfolio_ret_srs


def portfolios_linear(
        tids: list[str],
        bgn_date: str, stp_date: str | None,
        features_and_return_dir: str,
        group_tests_summary_dir: str,
        portfolios_dir: str,
        sqlite3_tables: dict,
):
    pool = mp.Pool(processes=5)
    res = []
    for tid in tids:
        print("...", "@", dt.datetime.now(), tid, "started")
        t = pool.apply_async(
            portfolios_linear_per_tid,
            args=(tid,
                  bgn_date, stp_date,
                  features_and_return_dir,
                  group_tests_summary_dir,
                  sqlite3_tables)
        )
        res.append(t)

    tid_ret_data = {}
    for tid, _ in zip(tids, res):
        tid_ret_data[tid] = _.get()
    tid_ret_df = pd.DataFrame(tid_ret_data).fillna(0)
    tid_ret_df_cumsum = tid_ret_df.cumsum()

    plot_lines(t_plot_df=tid_ret_df_cumsum,
               t_fig_name="portfolios-linear-nav",
               t_colormap="jet",
               t_save_dir=portfolios_dir)

    tid_ret_df.to_csv(
        os.path.join(portfolios_dir, "portfolios-linear-ret.csv.gz"),
        index_label="trade_date", float_format="%.8f"
    )
    tid_ret_df_cumsum.to_csv(
        os.path.join(portfolios_dir, "portfolios-linear-nav.csv.gz"),
        index_label="trade_date", float_format="%.8f"
    )

    ret_mu = tid_ret_df.mean()
    ret_sd = tid_ret_df.std()
    ret_sharpe = ret_mu / ret_sd * np.sqrt(252)
    performance_df = pd.DataFrame({
        "Aver": ret_mu,
        "Std": ret_sd,
        "Sharpe": ret_sharpe
    }).T
    performance_df.to_csv(
        os.path.join(portfolios_dir, "portfolios-linear-eval.csv"),
        float_format="%.6f"
    )

    pd.set_option("display.width", 0)
    print("\n", tid_ret_df)
    print("\n", tid_ret_df_cumsum)
    print("\n", performance_df)

    return 0
