from project_setup import calendar_path
from project_setup import futures_instru_info_path
from project_setup import equity_index_by_instrument_dir, md_by_instru_dir
from project_setup import futures_md_structure_path, futures_em01_db_name, futures_md_dir
from project_setup import major_minor_dir
from project_setup import research_features_and_return_dir
from project_setup import research_ic_tests_dir
from project_setup import research_ic_tests_summary_dir
from project_config import sqlite3_tables
from project_config import equity_indexes
from project_config import factors
from project_config import tids
from dp_00_features_and_return import split_spot_daily_k, cal_features_and_return
from dp_01_convert_csv_to_sqlite3 import convert_csv_to_sqlite3
from ic_tests import multi_process_fun_for_ic_tests
from ic_tests import ic_tests_summary

if __name__ == "__main__":

    md_bgn_date, md_stp_date = "20160101", "20230529"

    switch = {
        "split": False,
        "features_and_return": False,
        "toSql": False,
        "ic_tests": False,
        "ic_tests_summary": False,
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
