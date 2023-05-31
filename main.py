from project_setup import calendar_path
from project_setup import futures_instru_info_path
from project_setup import equity_index_by_instrument_dir, md_by_instru_dir
from project_setup import futures_md_structure_path, futures_em01_db_name, futures_md_dir
from project_setup import major_minor_dir
from project_setup import research_features_and_return_dir
from project_setup import research_ic_tests_dir
from project_setup import research_ic_tests_summary_dir
from project_setup import research_group_tests_dir
from project_setup import research_group_tests_summary_dir
from project_setup import research_portfolios_dir
from project_setup import research_models_dir
from project_config import sqlite3_tables
from project_config import equity_indexes
from project_config import factors
from project_config import instruments_universe, tids
from project_config import train_windows
from project_config import x_lbls, y_lbls
from project_config import cost_rate
from dp_00_features_and_return import split_spot_daily_k, cal_features_and_return
from dp_01_convert_csv_to_sqlite3 import convert_csv_to_sqlite3
from ic_tests import multi_process_fun_for_ic_tests
from ic_tests import ic_tests_summary
from group_tests import multi_process_fun_for_group_tests
from group_tests import group_tests_summary
from portfolios_linear import portfolios_linear_mp
from ml_normalize import ml_normalize_mp
from ml_train_rrcv import ml_rrcv_mp
from ml_train_mlpc import ml_mlpc_mp

if __name__ == "__main__":

    md_bgn_date, md_stp_date = "20160101", "20230529"
    trn_bgn_date, trn_stp_date = "20170101", "20230522"

    switch = {
        "split": False,
        "features_and_return": False,
        "toSql": False,
        "ic_tests": False,
        "ic_tests_summary": False,
        "group_tests": False,
        "group_tests_summary": False,
        "portfolios_linear": False,
        "normalize": False,
        "rrcv": False,
        "mlpc": False,
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
            research_features_and_return_dir=research_features_and_return_dir,
            verbose=False,
        )

    if switch["toSql"]:
        convert_csv_to_sqlite3(
            run_mode="o", bgn_date=md_bgn_date, stp_date=md_stp_date,
            calendar_path=calendar_path,
            research_features_and_return_dir=research_features_and_return_dir,
            equity_indexes=equity_indexes,
            sqlite3_tables=sqlite3_tables,
        )

    if switch["ic_tests"]:
        multi_process_fun_for_ic_tests(
            group_n=5,
            factors=factors, tids=tids,
            run_mode="o", bgn_date=md_bgn_date, stp_date=md_stp_date,
            features_and_return_dir=research_features_and_return_dir,
            ic_tests_dir=research_ic_tests_dir,
            sqlite3_tables=sqlite3_tables,
        )

    if switch["ic_tests_summary"]:
        ic_tests_summary(
            factors=factors, tids=tids,
            bgn_date=md_bgn_date, stp_date=md_stp_date,
            ic_tests_dir=research_ic_tests_dir,
            ic_tests_summary_dir=research_ic_tests_summary_dir,
            sqlite3_tables=sqlite3_tables
        )

    if switch["group_tests"]:
        multi_process_fun_for_group_tests(
            group_n=5,
            factors=factors, tids=tids,
            run_mode="o", bgn_date=md_bgn_date, stp_date=md_stp_date,
            features_and_return_dir=research_features_and_return_dir,
            group_tests_dir=research_group_tests_dir,
            sqlite3_tables=sqlite3_tables,
        )

    if switch["group_tests_summary"]:
        group_tests_summary(
            factors=factors, tids=tids,
            bgn_date=md_bgn_date, stp_date=md_stp_date,
            group_tests_dir=research_group_tests_dir,
            group_tests_summary_dir=research_group_tests_summary_dir,
            sqlite3_tables=sqlite3_tables
        )

    if switch["portfolios_linear"]:
        portfolios_linear_mp(
            proc_num=5,
            tids=tids, bgn_date=md_bgn_date, stp_date=md_stp_date,
            features_and_return_dir=research_features_and_return_dir,
            group_tests_summary_dir=research_group_tests_summary_dir,
            portfolios_dir=research_portfolios_dir,
            sqlite3_tables=sqlite3_tables
        )

    if switch["normalize"]:
        ml_normalize_mp(
            proc_num=5,
            instruments=instruments_universe + [None], tids=tids, train_windows=train_windows,
            bgn_date=trn_bgn_date, stp_date=trn_stp_date,
            calendar_path=calendar_path,
            features_and_return_dir=research_features_and_return_dir,
            models_dir=research_models_dir,
            sqlite3_tables=sqlite3_tables,
            x_lbls=x_lbls, y_lbls=y_lbls
        )

    if switch["rrcv"]:
        ml_rrcv_mp(
            proc_num=5,
            instruments=instruments_universe + [None], tids=tids, train_windows=train_windows,
            bgn_date=trn_bgn_date, stp_date=trn_stp_date,
            calendar_path=calendar_path,
            features_and_return_dir=research_features_and_return_dir,
            models_dir=research_models_dir,
            sqlite3_tables=sqlite3_tables,
            x_lbls=x_lbls, y_lbls=y_lbls
        )

    if switch["mlpc"]:
        ml_mlpc_mp(
            proc_num=5,
            instruments=instruments_universe + [None], tids=tids, train_windows=train_windows,
            bgn_date=trn_bgn_date, stp_date=trn_stp_date,
            calendar_path=calendar_path,
            features_and_return_dir=research_features_and_return_dir,
            models_dir=research_models_dir,
            sqlite3_tables=sqlite3_tables,
            x_lbls=x_lbls, y_lbls=y_lbls
        )
