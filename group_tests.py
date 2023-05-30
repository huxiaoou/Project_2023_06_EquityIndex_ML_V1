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
