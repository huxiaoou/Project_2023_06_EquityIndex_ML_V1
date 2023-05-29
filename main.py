import subprocess as sp
import itertools as ittl
from project_setup import calendar_path
from project_setup import futures_instru_info_path
from project_setup import equity_index_by_instrument_dir, md_by_instru_dir
from project_setup import futures_md_structure_path, futures_em01_db_name, futures_md_dir
from project_setup import major_minor_dir
from project_setup import research_features_and_return_dir
# from project_setup import research_models_dir
# from project_setup import research_navs_dir
# from project_setup import research_predictions_dir
# from project_setup import research_summary_dir
from project_config import sqlite3_tables
from project_config import equity_indexes
from project_config import instruments_universe, tids
# from project_config import train_windows
from project_config import x_lbls, y_lbls
from project_config import cost_rate
from dp_00_features_and_return import split_spot_daily_k, cal_features_and_return
from dp_01_convert_csv_to_sqlite3 import convert_csv_to_sqlite3

# from ml_normalize import ml_normalize
# from ml_train_lm import ml_lm
# from ml_train_mlpr import multi_process_fun_for_ml_mlpr
# from ml_train_mlpc import multi_process_fun_for_ml_mlpc
# from ml_test import multi_process_fun_for_ml_test
# from ml_summary import ml_summary

if __name__ == "__main__":

    md_bgn_date, md_stp_date = "20160101", "20230529"
    trn_bgn_date, trn_stp_date = "20180101", "20230529"

    switch = {
        "split": False,
        "features_and_return": False,
        "toSql": False,
        "normalize": False,
        "lm": False,
        "mlpr": False,
        "mlpc": False,
        "test": False,
        "summary": False,
    }

    if switch["split"]:
        split_spot_daily_k(equity_index_by_instrument_dir, equity_indexes)

    if switch["features_and_return"]:
        cal_features_and_return(
            bgn_date=md_bgn_date, stp_date=md_stp_date, equity_indexes=equity_indexes,
            calendar_path=calendar_path, futures_instru_info_path=futures_instru_info_path,
            equity_index_by_instrument_dir=equity_index_by_instrument_dir,
            md_by_instru_dir=md_by_instru_dir,
            futures_md_structure_path=futures_md_structure_path,
            futures_em01_db_name=futures_em01_db_name,
            futures_md_dir=futures_md_dir,
            major_minor_dir=major_minor_dir,
            research_features_and_return_dir=research_features_and_return_dir
        )

    if switch["toSql"]:
        convert_csv_to_sqlite3(
            run_mode="o", bgn_date=md_bgn_date, stp_date=md_stp_date,
            calendar_path=calendar_path,
            research_features_and_return_dir=research_features_and_return_dir,
            equity_indexes=equity_indexes,
            sqlite3_tables=sqlite3_tables
        )

    # if switch["normalize"]:
    #     for instrument, tid, trn_win in ittl.product(instruments_universe + [None], tids + [None], train_windows):
    #         ml_normalize(
    #             instrument=instrument, tid=tid, trn_win=trn_win,
    #             bgn_date=trn_bgn_date, stp_date=trn_stp_date,
    #             calendar_path=calendar_path,
    #             features_and_return_dir=research_features_and_return_dir,
    #             models_dir=research_models_dir,
    #             sqlite3_tables=sqlite3_tables,
    #             x_lbls=x_lbls, y_lbls=y_lbls
    #         )
    #
    # if switch["lm"]:
    #     for instrument, tid, trn_win in ittl.product(instruments_universe + [None], tids + [None], train_windows):
    #         ml_lm(
    #             instrument=instrument, tid=tid, trn_win=trn_win,
    #             bgn_date=trn_bgn_date, stp_date=trn_stp_date,
    #             calendar_path=calendar_path,
    #             features_and_return_dir=research_features_and_return_dir,
    #             models_dir=research_models_dir,
    #             sqlite3_tables=sqlite3_tables,
    #             x_lbls=x_lbls, y_lbls=y_lbls
    #         )
    #
    # if switch["mlpr"]:
    #     multi_process_fun_for_ml_mlpr(
    #         group_n=5,
    #         instruments=instruments_universe + [None], tids=tids + [None], train_windows=train_windows,
    #         bgn_date=trn_bgn_date, stp_date=trn_stp_date,
    #         calendar_path=calendar_path,
    #         features_and_return_dir=research_features_and_return_dir,
    #         models_dir=research_models_dir,
    #         sqlite3_tables=sqlite3_tables,
    #         x_lbls=x_lbls, y_lbls=y_lbls
    #     )
    #
    # if switch["mlpc"]:
    #     multi_process_fun_for_ml_mlpc(
    #         group_n=5,
    #         instruments=instruments_universe + [None], tids=tids + [None], train_windows=train_windows,
    #         bgn_date=trn_bgn_date, stp_date=trn_stp_date,
    #         calendar_path=calendar_path,
    #         features_and_return_dir=research_features_and_return_dir,
    #         models_dir=research_models_dir,
    #         sqlite3_tables=sqlite3_tables,
    #         x_lbls=x_lbls, y_lbls=y_lbls
    #     )
    #
    # if switch["test"]:
    #     multi_process_fun_for_ml_test(
    #         group_n=5,
    #         model_lbls=["lm", "mlpr", "mlpc"],
    #         instruments=instruments_universe + [None], tids=tids + [None], train_windows=train_windows,
    #         bgn_date=trn_bgn_date, stp_date=trn_stp_date,
    #         calendar_path=calendar_path,
    #         features_and_return_dir=research_features_and_return_dir,
    #         models_dir=research_models_dir,
    #         predictions_dir=research_predictions_dir,
    #         sqlite3_tables=sqlite3_tables,
    #         x_lbls=x_lbls, y_lbls=y_lbls
    #     )
    #
    # if switch["summary"]:
    #     for model_lbl in ["lm", "mlpr", "mlpc"]:
    #         ml_summary(
    #             model_lbl=model_lbl,
    #             instruments_universe=instruments_universe, tids=tids, train_windows=train_windows,
    #             sqlite3_tables=sqlite3_tables,
    #             predictions_dir=research_predictions_dir, navs_dir=research_navs_dir,
    #             research_summary_dir=research_summary_dir,
    #             cost_rate=cost_rate
    #         )
