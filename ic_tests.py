import os
import datetime as dt
import numpy as np
import pandas as pd
import itertools as ittl
import multiprocessing as mp
from skyrim.falkreath import CManagerLibReader, CTable, CManagerLibWriter
from skyrim.winterhold import plot_lines


def ic_test_per_factor_and_tid(
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
    ic_tests_lib.update(t_update_df=ic_tests_df, t_using_index=True)

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
            ic_test_per_factor_and_tid(factor=factor, tid=tid, **kwargs)
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
    ic_data_by_fac, ic_data_by_tid = {f: {} for f in factors}, {t: {} for t in tids}
    for factor, tid in ittl.product(factors, tids):
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

        ic_srs = ic_df["ic"]
        ic_data_by_fac[factor][tid] = ic_srs
        ic_data_by_tid[tid][factor] = ic_srs

        ic_tests_summary_data.append({
            "factor": factor,
            "tid": tid,
            "obs": len(ic_df),
            "mean": ic_srs.mean(),
            "std": ic_srs.std(),
            "icir": ic_srs.mean() / ic_srs.std() * np.sqrt(252),
        })

    # plot by factor
    for f, f_data in ic_data_by_fac.items():
        ic_df_by_fac = pd.DataFrame(f_data)
        ic_df_by_fac_cumsum = ic_df_by_fac.cumsum()
        plot_lines(
            t_plot_df=ic_df_by_fac_cumsum,
            t_fig_name="{}-ic-cumsum".format(f),
            t_ylim=(-150, 150),
            t_colormap="jet",
            t_save_dir=ic_tests_summary_dir
        )
        print("...", f, "plotted")

    # plot by tid
    for t, t_data in ic_data_by_tid.items():
        ic_df_by_tid = pd.DataFrame(t_data)
        ic_df_by_tid_cumsum = ic_df_by_tid.cumsum()

        sorted_cumsum = ic_df_by_tid_cumsum.iloc[-1, :].sort_values(ascending=False)
        head_factors = sorted_cumsum.head(8).index.to_list()
        tail_factors = sorted_cumsum.tail(8).index.to_list()

        plot_lines(
            t_plot_df=ic_df_by_tid_cumsum[head_factors + tail_factors],
            t_fig_name="{}-ic-cumsum".format(t),
            t_colormap="jet",
            t_ylim=(-150, 150),
            t_save_dir=ic_tests_summary_dir
        )
        print("...", t, "plotted")

    # summary all
    summary_df = pd.DataFrame(ic_tests_summary_data).sort_values(by=["tid", "icir"], ascending=[True, False])
    pd.set_option("display.max_rows", 16)
    for tid, tid_summary_df in summary_df.groupby(by="tid"):
        tid_summary_file = "ic_summary-{}-{}-{}.csv".format(tid, bgn_date, stp_date)
        tid_summary_df.to_csv(
            os.path.join(ic_tests_summary_dir, tid_summary_file),
            index=False, float_format="%.4f"
        )
        tid_summary_file = "ic_summary-{}-latest.csv".format(tid)
        tid_summary_df.to_csv(
            os.path.join(ic_tests_summary_dir, tid_summary_file),
            index=False, float_format="%.4f"
        )
        print("\n-------------------\n...", tid, "ic summary")
        print(tid_summary_df)
    return 0
