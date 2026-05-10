from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import io
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_CSV = os.path.join(BASE_DIR, "predictions_latest.csv")
LIVE_CSV   = os.path.join(BASE_DIR, "live_data.csv")


def parse_time(df):
    """
    Accept two CSV formats:
      A) Single column  : time_jst  (with or without timezone suffix)
      B) Split columns  : time_jst_date + time_jst_hour
    Returns a Series of timezone-naive datetime (JST assumed).
    """
    if "time_jst" in df.columns:
        return pd.to_datetime(df["time_jst"], utc=False).dt.tz_localize(None)

    if "time_jst_date" in df.columns and "time_jst_hour" in df.columns:
        combined = df["time_jst_date"].astype(str) + " " + df["time_jst_hour"].astype(str)
        return pd.to_datetime(combined, format="mixed")

    raise ValueError("時刻列が見つかりません（time_jst または time_jst_date+time_jst_hour）")


def process_csv(df):
    df = df.copy()
    df["_time"] = parse_time(df)
    df = df.sort_values("_time").reset_index(drop=True)

    def to_iso(series):
        return series.dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()

    t = df["_time"]
    times    = to_iso(t)
    times_1h = to_iso(t + pd.Timedelta(hours=1))
    times_4h = to_iso(t + pd.Timedelta(hours=4))
    times_8h = to_iso(t + pd.Timedelta(hours=8))

    latest       = df.iloc[-1]
    latest_time  = t.iloc[-1].strftime("%Y-%m-%d %H:%M JST")

    # is_valid: handle bool, 1/0, "True"/"False"
    def to_bool_str(val):
        if isinstance(val, bool):
            return str(val)
        if str(val).strip() in ("1", "True", "true"):
            return "True"
        return "False"

    # ---- シグナル計算: pred_close_8h が close を上回るか下回るかで方向を決め、反転した足をシグナルとする ----
    signal_buy_t, signal_buy_p, signal_sell_t, signal_sell_p = [], [], [], []
    latest_signal = "FLAT"
    if "pred_r8" in df.columns:
        pred_r8  = pd.to_numeric(df["pred_r8"], errors="coerce").fillna(0)
        is_valid = pd.to_numeric(df.get("is_valid", pd.Series([1] * len(df))), errors="coerce").fillna(1)

        cur8  = np.sign(pred_r8)
        prev8 = cur8.shift(1).fillna(0)
        reversal = (cur8 != prev8) & (cur8 != 0) & (is_valid == 1)

        buy_mask  = reversal & (cur8 > 0)
        sell_mask = reversal & (cur8 < 0)

        signal_buy_t  = to_iso(t[buy_mask])
        signal_buy_p  = df["close"][buy_mask].round(3).tolist()
        signal_sell_t = to_iso(t[sell_mask])
        signal_sell_p = df["close"][sell_mask].round(3).tolist()

        last_dir = float(cur8.iloc[-1]) if len(cur8) else 0
        latest_signal = "BUY" if last_dir > 0 else ("SELL" if last_dir < 0 else "FLAT")

    # ease_score_r8: 0-100 の信頼スコア（-1 = N/A）
    ease_raw = pd.to_numeric(df.get("ease_score_r8", pd.Series([-1] * len(df))), errors="coerce").fillna(-1)
    latest_ease = int(ease_raw.iloc[-1]) if len(ease_raw) else -1

    # 感情スコア（列がなければ空リスト）
    def _sent(col):
        if col not in df.columns:
            return []
        return pd.to_numeric(df[col], errors="coerce").round(4).tolist()

    return {
        "time":          times,
        "time_1h":       times_1h,
        "time_4h":       times_4h,
        "time_8h":       times_8h,
        "close":         df["close"].round(3).tolist(),
        "pred_close_1h": df["pred_close_1h"].round(3).tolist(),
        "pred_close_4h": df["pred_close_4h"].round(3).tolist(),
        "pred_close_8h": df["pred_close_8h"].round(3).tolist(),
        "latest_time":   latest_time,
        "latest_regime": str(latest.get("regime", "-")),
        "latest_is_valid": to_bool_str(latest.get("is_valid", True)),
        "latest_signal": latest_signal,
        "latest_ease":     latest_ease,
        "sent_score_1h":   _sent("sent_score_1h"),
        "sent_score_6h":   _sent("sent_score_6h"),
        "sent_score_24h":  _sent("sent_score_24h"),
        "regime":          df["regime"].tolist() if "regime" in df.columns else [],
        "signal_buy_t":  signal_buy_t,
        "signal_buy_p":  signal_buy_p,
        "signal_sell_t": signal_sell_t,
        "signal_sell_p": signal_sell_p,
    }


@app.route("/")
def index():
    csv_path = LIVE_CSV if os.path.exists(LIVE_CSV) else SAMPLE_CSV
    df = pd.read_csv(csv_path)
    data = process_csv(df)
    return render_template("index.html", data=json.dumps(data))


@app.route("/update", methods=["POST"])
def update():
    try:
        content = request.get_data(as_text=True)
        df = pd.read_csv(io.StringIO(content))
        process_csv(df)  # validate
        df.to_csv(LIVE_CSV, index=False, encoding="utf-8")
        return jsonify({"status": "ok", "rows": len(df)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "ファイルがありません"}), 400
    file = request.files["file"]
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "CSVファイルを選択してください"}), 400
    try:
        content = file.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content))
        data = process_csv(df)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
