import os
import datetime as dt
import numpy as np
import pandas as pd
import itertools as ittl
import multiprocessing as mp
from skyrim.falkreath import CManagerLibReader, CTable, CManagerLibWriter
from skyrim.winterhold import plot_lines


def cal_group_return(df: pd.DataFrame, fac: str, ret: str, ret_scale: int):
    n = len(df)
    m = int(n / 2)
    d = n - 2 * m
    lng_raw_wgt = np.array([1] * m + [0] * d + [0] * m)
    srt_raw_wgt = np.array([0] * m + [0] * d + [1] * m)
    lng_wgt = lng_raw_wgt / np.abs(lng_raw_wgt).sum()
    srt_wgt = srt_raw_wgt / np.abs(srt_raw_wgt).sum()
    hdg_wgt = lng_wgt * 0.5 - srt_wgt * 0.5
    sig_df = df.sort_values(by=fac, ascending=False)
    ret_lng = sig_df[ret] @ lng_wgt / ret_scale
    ret_srt = sig_df[ret] @ srt_wgt / ret_scale
    ret_hdg = sig_df[ret] @ hdg_wgt / ret_scale
    return ret_lng, ret_srt, ret_hdg


def group_test_per_factor_and_tid(
        factor: str, tid: str,
        run_mode: str, bgn_date: str, stp_date: str | None,
        features_and_return_dir: str,
        group_tests_dir: str,
        sqlite3_tables: dict,
):
    if stp_date is None:
        stp_date = (dt.datetime.strptime(bgn_date, "%Y%m%d") + dt.timedelta(days=1)).strftime("%Y%m%d")

    # --- load lib reader
    features_and_return_lib = CManagerLibReader(
        t_db_save_dir=features_and_return_dir,
        t_db_name="features_and_return.db"
    )
    features_and_return_db_stru = sqlite3_tables["features_and_return"]
    features_and_return_tab = CTable(t_table_struct=features_and_return_db_stru)
    features_and_return_lib.set_default(features_and_return_tab.m_table_name)

    # --- load lib writer
    group_tests_lib_id = "{}-{}-group_tests".format(factor, tid)
    group_tests_lib = CManagerLibWriter(
        t_db_save_dir=group_tests_dir,
        t_db_name=group_tests_lib_id + ".db"
    )
    group_tests_lib_stru = sqlite3_tables[group_tests_lib_id]
    group_tests_lib_tab = CTable(t_table_struct=group_tests_lib_stru)
    group_tests_lib.initialize_table(t_table=group_tests_lib_tab, t_remove_existence=run_mode.upper() in ["O", "OVERWRITE"])

    src_df = features_and_return_lib.read_by_conditions(
        t_conditions=[
            ("trade_date", ">=", bgn_date),
            ("trade_date", "<", stp_date),
            ("tid", "=", tid),
        ],
        t_value_columns=["trade_date", "instrument", "tid", factor, "rtm"]
    )

    group_tests_srs = src_df.groupby(by="trade_date").apply(
        cal_group_return, fac=factor, ret="rtm", ret_scale=100)
    lng_ret_srs, srt_ret_srs, hdg_ret_srs = zip(*group_tests_srs)
    group_tests_df = pd.DataFrame({
        "lng": lng_ret_srs,
        "srt": srt_ret_srs,
        "hdg": hdg_ret_srs,
    }, index=group_tests_srs.index).fillna(0)
    group_tests_lib.update(t_update_df=group_tests_df, t_using_index=True)

    # close libs
    features_and_return_lib.close()
    group_tests_lib.close()
    return 0


def process_target_fun_for_group_tests(
        group_id: int, group_n: int,
        factors: list[str], tids: list[str],
        **kwargs
):
    for i, (factor, tid) in enumerate(ittl.product(factors, tids)):
        if i % group_n == group_id:
            group_test_per_factor_and_tid(factor=factor, tid=tid, **kwargs)
    return 0


def multi_process_fun_for_group_tests(
        group_n: int,
        factors: list[str], tids: list[str],
        **kwargs):
    to_join_list = []
    for group_id in range(group_n):
        t = mp.Process(
            target=process_target_fun_for_group_tests,
            args=(group_id, group_n, factors, tids),
            kwargs=kwargs)
        t.start()
        to_join_list.append(t)
    for t in to_join_list:
        t.join()
    return 0


def group_tests_summary(
        factors: list[str], tids: list[str],
        bgn_date: str, stp_date: str,
        group_tests_dir: str,
        group_tests_summary_dir: str,
        sqlite3_tables: dict,
):
    group_tests_summary_data = []
    group_data_by_fac, group_data_by_tid = {f: {} for f in factors}, {t: {} for t in tids}
    for factor, tid in ittl.product(factors, tids):
        # --- load lib writer
        group_tests_lib_id = "{}-{}-group_tests".format(factor, tid)
        group_tests_lib = CManagerLibReader(
            t_db_save_dir=group_tests_dir,
            t_db_name=group_tests_lib_id + ".db"
        )
        group_tests_lib_stru = sqlite3_tables[group_tests_lib_id]
        group_tests_lib_tab = CTable(t_table_struct=group_tests_lib_stru)
        group_tests_lib.set_default(t_default_table_name=group_tests_lib_tab.m_table_name)

        ret_df = group_tests_lib.read_by_conditions(
            t_conditions=[
                ("trade_date", ">=", bgn_date),
                ("trade_date", "<", stp_date),
            ],
            t_value_columns=["trade_date", "lng", "srt", "hdg"]
        ).set_index("trade_date")

        ret_srs = ret_df["hdg"]
        group_data_by_fac[factor][tid] = ret_srs
        group_data_by_tid[tid][factor] = ret_srs

        group_tests_summary_data.append({
            "factor": factor,
            "tid": tid,
            "obs": len(ret_df),
            "mean": ret_srs.mean(),
            "std": ret_srs.std(),
            "sharpe": ret_srs.mean() / ret_srs.std() * np.sqrt(252),
        })

    # plot by factor
    for f, f_data in group_data_by_fac.items():
        ret_df_by_fac = pd.DataFrame(f_data)
        ret_df_by_fac_cumsum = ret_df_by_fac.cumsum()
        plot_lines(
            t_plot_df=ret_df_by_fac_cumsum,
            t_fig_name="{}-hdg-cumsum".format(f),
            # t_ylim=(-150, 150),
            t_colormap="jet",
            t_save_dir=group_tests_summary_dir
        )
        print("...", f, "plotted")

    # plot by tid
    for t, t_data in group_data_by_tid.items():
        ret_df_by_tid = pd.DataFrame(t_data)
        ret_df_by_tid_cumsum = ret_df_by_tid.cumsum()

        sorted_cumsum = ret_df_by_tid_cumsum.iloc[-1, :].sort_values(ascending=False)
        head_factors = sorted_cumsum.head(8).index.to_list()
        tail_factors = sorted_cumsum.tail(8).index.to_list()

        plot_lines(
            t_plot_df=ret_df_by_tid_cumsum[head_factors + tail_factors],
            t_fig_name="{}-hdg-cumsum".format(t),
            t_colormap="jet",
            # t_ylim=(-150, 150),
            t_save_dir=group_tests_summary_dir
        )
        print("...", t, "plotted")

        selected_factors_df = pd.DataFrame({
            "factors": head_factors + tail_factors,
            "direction": [1] * len(head_factors) + [-1] * len(tail_factors)
        })
        selected_factors_df.to_csv(
            os.path.join(group_tests_summary_dir, "selected-factors-{}.csv.gz".format(t)),
            index=False
        )

    # summary all
    summary_df = pd.DataFrame(group_tests_summary_data).sort_values(by=["tid", "sharpe"], ascending=[True, False])
    pd.set_option("display.max_rows", 16)
    for tid, tid_summary_df in summary_df.groupby(by="tid"):
        tid_summary_file = "hdg_summary-{}-{}-{}.csv".format(tid, bgn_date, stp_date)
        tid_summary_df.to_csv(
            os.path.join(group_tests_summary_dir, tid_summary_file),
            index=False, float_format="%.4f"
        )
        tid_summary_file = "hdg_summary-{}-latest.csv".format(tid)
        tid_summary_df.to_csv(
            os.path.join(group_tests_summary_dir, tid_summary_file),
            index=False, float_format="%.4f"
        )
        print("\n-------------------\n...", tid, "hdg summary")
        print(tid_summary_df)
    return 0
