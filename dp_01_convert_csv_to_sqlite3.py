import os
import datetime as dt
import pandas as pd
from skyrim.falkreath import CManagerLibWriterByDate, CTable
from skyrim.whiterun import CCalendar


def convert_csv_to_sqlite3(run_mode: str, bgn_date: str, stp_date: str,
                           calendar_path: str,
                           research_features_and_return_dir: str,
                           equity_indexes,
                           sqlite3_tables,
                           ):
    """

    :param run_mode: must be one of ['o', 'overwrite', 'a', 'append']
    :param bgn_date: begin date, format = [YYYYMMDD]
    :param stp_date: stop date, format = [YYYYMMDD], can be skip, and program will use bgn only
    :param calendar_path:
    :param research_features_and_return_dir:
    :param equity_indexes:
    :param sqlite3_tables:
    :return:
    """
    if stp_date is None:
        stp_date = (dt.datetime.strptime(bgn_date, "%Y%m%d") + dt.timedelta(days=1)).strftime("%Y%m%d")

    # --- load calendar
    calendar = CCalendar(calendar_path)

    # --- load lib writer
    features_and_return_lib = CManagerLibWriterByDate(
        t_db_save_dir=research_features_and_return_dir,
        t_db_name="features_and_return.db"
    )
    features_and_return_tab = CTable(t_table_struct=sqlite3_tables["features_and_return"])
    features_and_return_lib.initialize_table(
        t_table=features_and_return_tab,
        t_remove_existence=run_mode.upper() in ["O", "OVERWRITE"]
    )

    for trade_date in calendar.get_iter_list(bgn_date, stp_date, True):
        save_date_dir = os.path.join(research_features_and_return_dir, trade_date[0:4], trade_date)
        for equity_index_code, equity_instru_id in equity_indexes:
            if trade_date <= "20220722" and equity_instru_id == "IM.CFE":
                continue

            features_and_return_file = "{}-{}-features_and_return.csv.gz".format(trade_date, equity_instru_id)
            features_and_return_path = os.path.join(save_date_dir, features_and_return_file)
            features_and_ret_df = pd.read_csv(features_and_return_path, dtype={"trade_date": str, "timestamp": int})

            if run_mode in ["A", "APPEND"]:
                print(features_and_ret_df)
                features_and_return_lib.delete_by_date(t_date=trade_date)
            features_and_return_lib.update_by_date(
                t_date=trade_date,
                t_update_df=features_and_ret_df,
            )

        print("... @ {0}, features and return of {1} converted to sqlite3".format(dt.datetime.now(), trade_date))

    features_and_return_lib.close()
    return 0
