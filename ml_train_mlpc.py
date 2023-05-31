import os
import datetime as dt
import itertools as ittl
import multiprocessing as mp
import numpy as np
from sklearn.neural_network import MLPClassifier
from skyrim.falkreath import CManagerLibReader, CTable
from skyrim.whiterun import CCalendarMonthly
from xfuns import save_to_sio_obj
from xfuns import read_from_sio_obj


def ml_mlpc_per_instru_and_tid(
        instrument: str | None, tid: str, trn_win: int,
        bgn_date: str, stp_date: str,
        calendar_path: str,
        features_and_return_dir: str, models_dir: str,
        sqlite3_tables: dict,
        x_lbls: list, y_lbls: list,
):
    """

    :param instrument: like IC.CFE, and None means all the data
    :param tid: ['T01',...,'T07']
    :param trn_win: [12,24,36]
    :param bgn_date: format = [YYYYMMDD]
    :param stp_date: format = [YYYYMMDD], can be skip, and program will use bgn only
    :param calendar_path:
    :param features_and_return_dir:
    :param models_dir:
    :param sqlite3_tables:
    :param x_lbls:
    :param y_lbls: "rtm" must be in it
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
    train_model, model_lbl = \
        MLPClassifier(hidden_layer_sizes=(5, 5),
                      solver="adam", random_state=0, alpha=1.0, max_iter=2000), "mlpc"
    for train_end_month in iter_months:
        model_year_dir = os.path.join(models_dir, train_end_month[0:4])
        model_month_dir = os.path.join(model_year_dir, train_end_month)

        train_bgn_date, train_end_date = calendar.get_bgn_and_end_dates_for_trailing_window(train_end_month, trn_win)
        conds = init_conds + [
            ("trade_date", ">=", train_bgn_date),
            ("trade_date", "<=", train_end_date),
        ]
        src_df = features_and_return_lib.read_by_conditions(
            t_conditions=conds,
            t_value_columns=x_lbls + y_lbls
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
        x_train = np.nan_to_num(scaler.transform(x_df), nan=0)
        train_model.fit(X=x_train, y=[1 if z >= 0 else 0 for z in y_df.values[:, 0]])

        train_model_file = "{}-{}.{}".format(model_grp_id, train_end_month, model_lbl)
        train_model_path = os.path.join(model_month_dir, train_model_file)

        # --- save model
        save_to_sio_obj(train_model, train_model_path)

        print("... {0} | {1} | {2:>24s} | {3} | fitted |".format(
            dt.datetime.now(), model_lbl, model_grp_id, train_end_month))

    features_and_return_lib.close()
    return 0


def ml_mlpc_mp(proc_num: int,
               instruments: list[str | None], tids: list[str], train_windows: list[int],
               **kwargs):
    t0 = dt.datetime.now()
    pool = mp.Pool(processes=proc_num)
    for instrument, tid, trn_win in ittl.product(instruments, tids, train_windows):
        pool.apply_async(
            ml_mlpc_per_instru_and_tid,
            args=(instrument, tid, trn_win),
            kwds=kwargs
        )
    pool.close()
    pool.join()
    t1 = dt.datetime.now()
    print("... total time consuming: {:.2f} seconds".format((t1 - t0).total_seconds()))
    return 0
