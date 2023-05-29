import os
import datetime as dt
import numpy as np
import pandas as pd
import itertools as ittl
import multiprocessing as mp
from skyrim.falkreath import CManagerLibReader, CTable, CManagerLibWriter
from skyrim.winterhold import plot_lines


def it_test_per_factor_and_tid(
        factor: str, tid: str,
        run_mode: str, bgn_date: str, stp_date: str | None,
        features_and_return_dir: str,
        ic_tests_dir: str,
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
    ic_tests_lib_id = "{}-{}-ic_tests".format(factor, tid)
    ic_tests_lib = CManagerLibWriter(
        t_db_save_dir=ic_tests_dir,
        t_db_name=ic_tests_lib_id + ".db"
    )
    ic_tests_lib_stru = sqlite3_tables[ic_tests_lib_id]
    ic_tests_lib_tab = CTable(t_table_struct=ic_tests_lib_stru)
    ic_tests_lib.initialize_table(t_table=ic_tests_lib_tab, t_remove_existence=run_mode.upper() in ["O", "OVERWRITE"])

    src_df = features_and_return_lib.read_by_conditions(
        t_conditions=[
            ("trade_date", ">=", bgn_date),
            ("trade_date", "<", stp_date),
            ("tid", "=", tid),
        ],
        t_value_columns=["trade_date", "instrument", "tid", factor, "rtm"]
    )

    ic_tests_srs = src_df.groupby(by="trade_date").apply(
        lambda z: (len(z), z[[factor, "rtm"]].corr(method="spearman").at[factor, "rtm"]))
    obs_srs, ic_srs = zip(*ic_tests_srs)
    ic_tests_df = pd.DataFrame({
        "obs": obs_srs,
        "ic": ic_srs
    }, index=ic_tests_srs.index).fillna(0)
    ic_tests_lib.update(t_update_df=ic_tests_df, t_using_index=True, )

    # close libs
    features_and_return_lib.close()
    ic_tests_lib.close()
    return 0


def process_target_fun_for_ic_tests(
        group_id: int, group_n: int,
        factors: list[str], tids: list[str],
        **kwargs
):
    for i, (factor, tid) in enumerate(ittl.product(factors, tids)):
        if i % group_n == group_id:
            it_test_per_factor_and_tid(factor=factor, tid=tid, **kwargs)
    return 0


def multi_process_fun_for_ic_tests(
        group_n: int,
        factors: list[str], tids: list[str],
        **kwargs):
    to_join_list = []
    for group_id in range(group_n):
        t = mp.Process(
            target=process_target_fun_for_ic_tests,
            args=(group_id, group_n, factors, tids),
            kwargs=kwargs)
        t.start()
        to_join_list.append(t)
    for t in to_join_list:
        t.join()
    return 0


def ic_tests_summary(
        factors: list[str], tids: list[str],
        bgn_date: str, stp_date: str,
        ic_tests_dir: str,
        ic_tests_summary_dir: str,
        sqlite3_tables: dict,
):
    ic_tests_summary_data = []
    for factor in factors:
        factor_ic_data = {}
        for tid in tids:
            # --- load lib writer
            ic_tests_lib_id = "{}-{}-ic_tests".format(factor, tid)
            ic_tests_lib = CManagerLibReader(
                t_db_save_dir=ic_tests_dir,
                t_db_name=ic_tests_lib_id + ".db"
            )
            ic_tests_lib_stru = sqlite3_tables[ic_tests_lib_id]
            ic_tests_lib_tab = CTable(t_table_struct=ic_tests_lib_stru)
            ic_tests_lib.set_default(t_default_table_name=ic_tests_lib_tab.m_table_name)

            ic_df = ic_tests_lib.read_by_conditions(
                t_conditions=[
                    ("trade_date", ">=", bgn_date),
                    ("trade_date", "<", stp_date),
                ],
                t_value_columns=["trade_date", "ic"]
            ).set_index("trade_date")
            factor_ic_data[tid] = ic_df["ic"]

            ic_tests_summary_data.append({
                "factor": factor,
                "tid": tid,
                "obs": len(ic_df),
                "mean": ic_df["ic"].mean(),
                "std": ic_df["ic"].std(),
                "icir": ic_df["ic"].mean() / ic_df["ic"].std() * np.sqrt(252),
            })

        factor_ic_df = pd.DataFrame(factor_ic_data)
        factor_ic_df_cumsum = factor_ic_df.cumsum()
        plot_lines(
            t_plot_df=factor_ic_df_cumsum,
            t_fig_name="{}-ic-cumsum".format(factor),
            t_colormap="jet",
            t_save_dir=ic_tests_summary_dir
        )

    summary_df = pd.DataFrame(ic_tests_summary_data)
    summary_file = "ic_summary-{}-{}.csv".format(bgn_date, stp_date)
    summary_df.to_csv(
        os.path.join(ic_tests_summary_dir, summary_file),
        index=False, float_format="%.4f"
    )
    summary_file = "ic_summary-latest.csv".format(bgn_date, stp_date)
    summary_df.to_csv(
        os.path.join(ic_tests_summary_dir, summary_file),
        index=False, float_format="%.4f"
    )
    return 0
