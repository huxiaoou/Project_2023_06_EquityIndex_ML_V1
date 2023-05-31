import os
import datetime as dt
import itertools as ittl
import multiprocessing as mp
import numpy as np
from skyrim.falkreath import CManagerLibReader, CManagerLibWriter, CTable
from skyrim.whiterun import CCalendarMonthly
from xfuns import read_from_sio_obj


def ml_test_per_model(
        model_lbl: str,
        instrument: str | None, tid: str, trn_win: int,
        bgn_date: str, stp_date: str,
        calendar_path: str,
        features_and_return_dir: str, models_dir: str, predictions_dir: str,
        sqlite3_tables: dict,
        x_lbls: list, y_lbls: list,
):
    """

    :param model_lbl: ["rrcv", "mlpc"]
    :param instrument: like IC.CFE, and None means all the data
    :param tid: ['T01',...,'T07']
    :param trn_win: [12,24,36]
    :param bgn_date: format = [YYYYMMDD]
    :param stp_date: format = [YYYYMMDD], can be skip, and program will use bgn only
    :param calendar_path:
    :param features_and_return_dir:
    :param models_dir:
    :param predictions_dir：
    :param sqlite3_tables:
    :param x_lbls:
    :param y_lbls: "rtm" must be in it
    :return:
    """

    init_conds = [(k, "=", v) for k, v in zip(("instrument", "tid"), (instrument, tid)) if v is not None]
    model_grp_id = "-".join(filter(lambda z: z, ["M", instrument, tid, "TMW{:02d}".format(trn_win)]))
    pred_id = model_grp_id + "-pred-{}".format(model_lbl)
    pred_header_cols = ["trade_date", "instrument", "contract", "tid", "timestamp"]

    if stp_date is None:
        stp_date = (dt.datetime.strptime(bgn_date, "%Y%m%d") + dt.timedelta(days=1)).strftime("%Y%m%d")

    # --- load calendar
    calendar = CCalendarMonthly(calendar_path)

    # --- load lib reader
    features_and_return_lib = CManagerLibReader(
        t_db_save_dir=features_and_return_dir,
        t_db_name="features_and_return.db"
    )
    features_and_return_db_stru = sqlite3_tables["features_and_return"]
    features_and_return_tab = CTable(t_table_struct=features_and_return_db_stru)
    features_and_return_lib.set_default(features_and_return_tab.m_table_name)

    # --- load lib writer
    predictions_lib = CManagerLibWriter(
        t_db_save_dir=predictions_dir,
        t_db_name=pred_id + ".db",
    )
    predictions_lib_stru = sqlite3_tables[pred_id]
    predictions_lib_tab = CTable(t_table_struct=predictions_lib_stru)
    predictions_lib.initialize_table(predictions_lib_tab)

    # --- dates
    iter_months = calendar.map_iter_dates_to_iter_months(bgn_date, stp_date)

    # --- main core
    for train_end_month in iter_months:
        model_year_dir = os.path.join(models_dir, train_end_month[0:4])
        model_month_dir = os.path.join(model_year_dir, train_end_month)

        test_month = calendar.get_next_month(train_end_month, 1)

        test_bgn_date, test_end_date = calendar.get_first_date_of_month(test_month), calendar.get_last_date_of_month(test_month)
        conds = init_conds + [
            ("trade_date", ">=", test_bgn_date),
            ("trade_date", "<=", test_end_date),
        ]
        src_df = features_and_return_lib.read_by_conditions(
            t_conditions=conds,
            t_value_columns=pred_header_cols + x_lbls + y_lbls
        )
        x_df, y_df = src_df[x_lbls], src_df[y_lbls]

        # --- normalize
        scaler_path = os.path.join(
            model_month_dir,
            "{}-{}.scl".format(model_grp_id, train_end_month)
        )
        try:
            scaler = read_from_sio_obj(scaler_path)
        except FileNotFoundError:
            continue

        # --- fit model
        x_test = np.nan_to_num(scaler.transform(x_df), nan=0)

        train_model_file = "{}-{}.{}".format(model_grp_id, train_end_month, model_lbl)
        train_model_path = os.path.join(model_month_dir, train_model_file)

        # --- load model
        train_model = read_from_sio_obj(train_model_path)

        # --- prediction
        src_df["pred"] = train_model.predict(X=x_test)
        predictions_lib.update(
            t_update_df=src_df[pred_header_cols + ["rtm", "pred"]],
            t_using_index=False
        )

    print("... {0} | {1} | {2:>24s} | {3} | tested |".format(
        dt.datetime.now(), model_lbl, model_grp_id, train_end_month))
    predictions_lib.close()
    features_and_return_lib.close()
    return 0


def ml_test_mp(proc_num: int,
               model_lbls: list[str],
               instruments: list[str | None], tids: list[str], train_windows: list[int],
               **kwargs):
    t0 = dt.datetime.now()
    pool = mp.Pool(processes=proc_num)
    for model_lbl, instrument, tid, trn_win in ittl.product(model_lbls, instruments, tids, train_windows):
        pool.apply_async(
            ml_test_per_model,
            args=(model_lbl, instrument, tid, trn_win),
            kwds=kwargs
        )
    pool.close()
    pool.join()
    t1 = dt.datetime.now()
    print("... total time consuming: {:.2f} seconds".format((t1 - t0).total_seconds()))
    return 0
