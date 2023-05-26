import sys
import os
import datetime as dt
import json
import pandas as pd
from skyrim.whiterun import CCalendar, CInstrumentInfoTable
from skyrim.winterhold import check_and_mkdir
from skyrim.falkreath import CManagerLibReader, CTable
from xfuns import cal_features_and_return_one_day


def split_spot_daily_k(equity_index_by_instrument_dir: str, equity_indexes: list[str]):
    daily_k_file = "daily_k.xlsx"
    daily_k_path = os.path.join(equity_index_by_instrument_dir, daily_k_file)
    for equity_index_code, _ in equity_indexes:
        daily_k_df = pd.read_excel(daily_k_path, sheet_name=equity_index_code)
        daily_k_df["trade_date"] = daily_k_df["trade_date"].map(lambda z: z.strftime("%Y%m%d"))
        equity_index_file = "{}.csv".format(equity_index_code)
        equity_index_path = os.path.join(equity_index_by_instrument_dir, equity_index_file)
        daily_k_df.to_csv(equity_index_path, index=False, float_format="%.4f")
    return 0


def cal_features_and_return(bgn_date: str, stp_date: str,
                            equity_indexes: list[str],
                            calendar_path: str, futures_instru_info_path: str,
                            equity_index_by_instrument_dir: str,
                            md_by_instru_dir: str,
                            futures_md_structure_path: str,
                            futures_em01_db_name: str,
                            futures_md_dir: str,
                            major_minor_dir: str,
                            research_features_and_return_dir: str,
                            verbose: bool = False
                            ):
    im_bgn_date = "20220722"
    id_cols = ["timestamp", "loc_id", "instrument", "exchange", "wind_code"]
    val_cols = [
        "open", "high", "low", "close",
        "volume", "amount", "oi",
        "daily_open", "daily_high", "daily_low",
        "preclose", "preoi",
    ]
    m01_columns = id_cols + val_cols

    # --- load calendar
    calendar = CCalendar(calendar_path)

    # --- load instru info table
    instru_info_table = CInstrumentInfoTable(t_path=futures_instru_info_path, t_index_label="windCode", t_type="CSV")

    # --- spot and futures manager
    spot_data_manager, futures_md_manager, major_minor_manager = {}, {}, {}
    for equity_index_code, equity_instru_id in equity_indexes:
        spot_data_file = "{}.csv".format(equity_index_code)
        spot_data_path = os.path.join(equity_index_by_instrument_dir, spot_data_file)
        spot_df = pd.read_csv(spot_data_path, dtype={"trade_date": str}).set_index("trade_date")
        spot_data_manager[equity_instru_id] = spot_df

        futures_md_file = "{}.md.settle.csv.gz".format(equity_instru_id)
        futures_md_path = os.path.join(md_by_instru_dir, futures_md_file)
        futures_md_df = pd.read_csv(futures_md_path, dtype={"trade_date": str}).set_index("trade_date")
        futures_md_manager[equity_instru_id] = futures_md_df

        major_minor_file = "major_minor.{}.csv.gz".format(equity_instru_id)
        major_minor_path = os.path.join(major_minor_dir, major_minor_file)
        major_minor_df = pd.read_csv(major_minor_path, dtype=str).set_index("trade_date")
        major_minor_manager[equity_instru_id] = major_minor_df

        print("... {}:{} spot and futures data loaded @ {}".format(equity_index_code, equity_instru_id, dt.datetime.now()))

    # --- init lib writer
    with open(futures_md_structure_path, "r") as j:
        m01_table_struct = json.load(j)[futures_em01_db_name]["CTable"]
    m01_table = CTable(t_table_struct=m01_table_struct)
    m01_db = CManagerLibReader(t_db_save_dir=futures_md_dir, t_db_name=futures_em01_db_name + ".db")
    m01_db.set_default(m01_table.m_table_name)

    # --- main loop
    for trade_date in calendar.get_iter_list(bgn_date, stp_date, True):
        prev_date = calendar.get_next_date(trade_date, -1)
        m01_df = m01_db.read_by_date(t_trade_date=trade_date, t_value_columns=m01_columns)
        if len(m01_df) == 0:
            continue

        check_and_mkdir(save_year_dir := os.path.join(research_features_and_return_dir, trade_date[0:4]))
        check_and_mkdir(save_date_dir := os.path.join(save_year_dir, trade_date))

        for equity_index_code, equity_instru_id in equity_indexes:
            if (trade_date <= im_bgn_date) and (equity_instru_id == "IM.CFE"):
                continue
            try:
                major_contract = major_minor_manager[equity_instru_id].at[trade_date, "n_contract"]
                pre_settle = futures_md_manager[equity_instru_id].at[prev_date, major_contract]
                pre_spot_close = spot_data_manager[equity_instru_id].at[prev_date, "close"]
            except KeyError:
                print(equity_instru_id, "does not have major contract @ ", trade_date)
                sys.exit()
            major_contract_m01_df = m01_df.loc[m01_df.wind_code == major_contract].reset_index(drop=True)
            if (num_of_bars := len(major_contract_m01_df)) != 240:
                print("Error! Number of bars = {} @ {} for {} - {}".format(
                    num_of_bars, trade_date, equity_instru_id, major_contract))
                sys.exit()

            contract_multiplier = instru_info_table.get_multiplier(equity_instru_id)
            features_and_ret_df = cal_features_and_return_one_day(
                m01=major_contract_m01_df,
                instrument=equity_instru_id, contract=major_contract, contract_multiplier=contract_multiplier,
                pre_settle=pre_settle, pre_spot_close=pre_spot_close)

            features_and_return_file = "{}-{}-features_and_return.csv.gz".format(trade_date, equity_instru_id)
            features_and_return_path = os.path.join(save_date_dir, features_and_return_file)
            features_and_ret_df.to_csv(features_and_return_path, index=False, float_format="%.6f")

        if verbose:
            print("... features and return are calculated for {}".format(trade_date))

    # --- close lib
    m01_db.close()

    return 0
