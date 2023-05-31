import os
import datetime as dt
import itertools as ittl
import multiprocessing as mp
from sklearn.preprocessing import StandardScaler
from skyrim.falkreath import CManagerLibReader, CTable
from skyrim.whiterun import CCalendarMonthly
from skyrim.winterhold import check_and_mkdir
from xfuns import save_to_sio_obj


def ml_normalize_per_instru_and_tid(
        instrument: str | None, tid: str, trn_win: int,
        bgn_date: str, stp_date: str,
        calendar_path: str,
        features_and_return_dir: str, models_dir: str,
        sqlite3_tables: dict,
        x_lbls: list, y_lbls: list,
        minimum_data_size: int = 100
):
    """

    :param instrument: like IC.CFE, and None means all the data
    :param tid: ['T01',...,'T07']
    :param trn_win: [6,12,24,36]
    :param bgn_date: format = [YYYYMMDD]
    :param stp_date: format = [YYYYMMDD], can be skip, and program will use bgn only
    :param calendar_path:
    :param features_and_return_dir:
    :param models_dir:
    :param sqlite3_tables:
    :param x_lbls:
    :param y_lbls: "rtm" must be in it
    :param minimum_data_size:
    :return:
    """

    init_conds = [(k, "=", v) for k, v in zip(("instrument", "tid"), (instrument, tid)) if v is not None]
    model_grp_id = "-".join(filter(lambda z: z, ["M", instrument, tid, "TMW{:02d}".format(trn_win)]))

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

    # --- dates
    iter_months = calendar.map_iter_dates_to_iter_months(bgn_date, stp_date)

    # --- main core
    scaler = StandardScaler()
    for train_end_month in iter_months:
        check_and_mkdir(model_year_dir := os.path.join(models_dir, train_end_month[0:4]))
        check_and_mkdir(model_month_dir := os.path.join(model_year_dir, train_end_month))

        train_bgn_date, train_end_date = calendar.get_bgn_and_end_dates_for_trailing_window(train_end_month, trn_win)
        conds = init_conds + [
            ("trade_date", ">=", train_bgn_date),
            ("trade_date", "<=", train_end_date),
        ]
        src_df = features_and_return_lib.read_by_conditions(
            t_conditions=conds,
            t_value_columns=x_lbls + y_lbls
        )
        if len(src_df) < minimum_data_size:
            continue
        x_df, y_df = src_df[x_lbls], src_df[y_lbls]

        # --- normalize
        scaler_path = os.path.join(
            model_month_dir,
            "{}-{}.scl".format(model_grp_id, train_end_month)
        )
        scaler.fit(x_df)
        save_to_sio_obj(scaler, scaler_path)

    print("... {0} | NORMALIZED | {1:>24s} | Normalized |".format(
        dt.datetime.now(), model_grp_id))

    features_and_return_lib.close()
    return 0


def ml_normalize_mp(proc_num: int,
                    instruments: list[str | None], tids: list[str], train_windows: list[int],
                    **kwargs):
    t0 = dt.datetime.now()
    pool = mp.Pool(processes=proc_num)
    for instrument, tid, trn_win in ittl.product(instruments, tids, train_windows):
        pool.apply_async(
            ml_normalize_per_instru_and_tid,
            args=(instrument, tid, trn_win),
            kwds=kwargs
        )
    pool.close()
    pool.join()
    t1 = dt.datetime.now()
    print("... total time consuming: {:.2f} seconds".format((t1 - t0).total_seconds()))
    return 0
