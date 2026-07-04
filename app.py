from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import io
import os
import tempfile

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_CSV = os.path.join(BASE_DIR, "predictions_latest.csv")
LIVE_CSV   = os.path.join(BASE_DIR, "live_data.csv")

# ForexLens 独立予想（YenCast の ML とは別系統。forex-lens 毎時batが POST する）
FL_FORECAST_CSV = os.path.join(BASE_DIR, "forexlens_forecast.csv")
FL_MOOD_CSV     = os.path.join(BASE_DIR, "forexlens_news_mood.csv")
FL_COT_CSV      = os.path.join(BASE_DIR, "forexlens_cot.csv")

# [W-1] /update は公開 URL なので、共有トークンを知る predictor 以外からの
# 書き込みを拒否する。Render の環境変数 YENCAST_UPDATE_TOKEN に設定する。
# 未設定なら認証なし（ローカル開発用）だが、本番では必ず設定すること。
UPDATE_TOKEN = os.environ.get("YENCAST_UPDATE_TOKEN", "").strip()

# [W-2] 表示・保持する最大行数。これを超える古い行は切り捨てる（描画と計算を軽く保つ）。
MAX_ROWS = int(os.environ.get("YENCAST_MAX_ROWS", "5000"))


def parse_time(df):
    """
    Accept two CSV formats:
      A) Single column  : time_jst  (with or without timezone suffix)
      B) Split columns  : time_jst_date + time_jst_hour
    Returns a Series of timezone-naive datetime (JST assumed).
    """
    if "time_jst" in df.columns:
        # [W-5] tz-aware/naive どちらでも落ちないようにする。
        # tz_localize(None) は tz-naive な Series に対して TypeError を投げるため、
        # まず utc=True でパースしてから naive 化する（混在入力にも耐える）。
        s = pd.to_datetime(df["time_jst"], errors="coerce", utc=True)
        return s.dt.tz_convert(None)

    if "time_jst_date" in df.columns and "time_jst_hour" in df.columns:
        combined = df["time_jst_date"].astype(str) + " " + df["time_jst_hour"].astype(str)
        return pd.to_datetime(combined, format="mixed")

    raise ValueError("時刻列が見つかりません（time_jst または time_jst_date+time_jst_hour）")


def _clean_nan(obj):
    """[NaN対策] JSON に NaN/Inf を出さないよう再帰的に None へ置換する。
    json.dumps / jsonify は既定で NaN リテラルを吐き、厳密 JSON パーサで壊れるため。
    """
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, list):
        return [_clean_nan(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    return obj


def process_csv(df):
    df = df.copy()
    df["_time"] = parse_time(df)
    df = df.dropna(subset=["_time"]).sort_values("_time").reset_index(drop=True)

    # [W-2] 古い行を切り捨てて最大行数を超えないようにする。
    if MAX_ROWS > 0 and len(df) > MAX_ROWS:
        df = df.tail(MAX_ROWS).reset_index(drop=True)

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

    # close は数値化しておく（文字列で来ても round で落ちないように）
    close_num = pd.to_numeric(df["close"], errors="coerce")

    # ---- シグナル計算: pred_r8 の符号が反転した足を BUY/SELL シグナルとする ----
    signal_buy_t, signal_buy_p, signal_sell_t, signal_sell_p = [], [], [], []
    latest_signal = "FLAT"
    if "pred_r8" in df.columns:
        pred_r8  = pd.to_numeric(df["pred_r8"], errors="coerce").fillna(0)
        is_valid = pd.to_numeric(df.get("is_valid", pd.Series([1] * len(df))), errors="coerce").fillna(1)

        cur8  = np.sign(pred_r8)
        prev8 = cur8.shift(1).fillna(0)
        # [W-7] invalid 区間明けの 1 本目に誤シグナルが出ないよう、
        # 今足だけでなく前足も valid であることを条件に加える。
        prev_valid = is_valid.shift(1).fillna(0)
        reversal = (cur8 != prev8) & (cur8 != 0) & (is_valid == 1) & (prev_valid == 1)

        buy_mask  = reversal & (cur8 > 0)
        sell_mask = reversal & (cur8 < 0)

        signal_buy_t  = to_iso(t[buy_mask])
        signal_buy_p  = close_num[buy_mask].round(3).tolist()
        signal_sell_t = to_iso(t[sell_mask])
        signal_sell_p = close_num[sell_mask].round(3).tolist()

        # 最新足の方向: invalid なら FLAT
        last_valid = float(is_valid.iloc[-1]) if len(is_valid) else 0
        last_dir = float(cur8.iloc[-1]) if len(cur8) else 0
        if last_valid != 1:
            latest_signal = "FLAT"
        else:
            latest_signal = "BUY" if last_dir > 0 else ("SELL" if last_dir < 0 else "FLAT")

    # ease_score_r8: 0-100 の信頼スコア（-1 = N/A）
    ease_raw = pd.to_numeric(df.get("ease_score_r8", pd.Series([-1] * len(df))), errors="coerce").fillna(-1)
    latest_ease = int(ease_raw.iloc[-1]) if len(ease_raw) else -1

    # 感情スコア（列がなければ空リスト）
    def _sent(col):
        if col not in df.columns:
            return []
        return pd.to_numeric(df[col], errors="coerce").round(4).tolist()

    def _num_list(col):
        return pd.to_numeric(df[col], errors="coerce").round(3).tolist()

    # ニュース地合い（参考・予測には不使用）。列が無ければ空リスト＝パネルは空表示。
    def _news_list(col):
        if col not in df.columns:
            return []
        return pd.to_numeric(df[col], errors="coerce").tolist()

    result = {
        "time":          times,
        "time_1h":       times_1h,
        "time_4h":       times_4h,
        "time_8h":       times_8h,
        "close":         close_num.round(3).tolist(),
        "pred_close_1h": _num_list("pred_close_1h"),
        "pred_close_4h": _num_list("pred_close_4h"),
        "pred_close_8h": _num_list("pred_close_8h"),
        "latest_time":   latest_time,
        "latest_regime": str(latest.get("regime", "-")),
        "latest_is_valid": to_bool_str(latest.get("is_valid", True)),
        "latest_signal": latest_signal,
        "latest_ease":     latest_ease,
        "sent_score_1h":   _sent("sent_score_1h"),
        "sent_score_6h":   _sent("sent_score_6h"),
        "sent_score_24h":  _sent("sent_score_24h"),
        "news_net":        _news_list("news_net"),
        "news_bull":       _news_list("news_bull"),
        "news_bear":       _news_list("news_bear"),
        "regime":          df["regime"].tolist() if "regime" in df.columns else [],
        "signal_buy_t":  signal_buy_t,
        "signal_buy_p":  signal_buy_p,
        "signal_sell_t": signal_sell_t,
        "signal_sell_p": signal_sell_p,
    }
    return _clean_nan(result)


def _atomic_write_csv(df, path):
    """[W-8] tmp に書いてから rename することで、書き込み途中の壊れた CSV を
    / ルートが読むのを防ぐ（rename は同一ディレクトリ内で原子的）。
    """
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


@app.route("/")
def index():
    # [W-6] CSV 欠損列などで落ちてもトップページが 500 にならないようにする。
    try:
        csv_path = LIVE_CSV if os.path.exists(LIVE_CSV) else SAMPLE_CSV
        df = pd.read_csv(csv_path)
        data = process_csv(df)
        return render_template("index.html", data=json.dumps(data))
    except Exception as e:
        # 最低限のエラーページ（空データ）を返す
        empty = json.dumps({"time": [], "error": str(e)})
        return render_template("index.html", data=empty)


@app.route("/update", methods=["POST"])
def update():
    # [W-1] トークン認証。設定されている場合のみ照合する。
    if UPDATE_TOKEN:
        auth = request.headers.get("Authorization", "")
        token = auth[7:].strip() if auth.startswith("Bearer ") else ""
        if token != UPDATE_TOKEN:
            return jsonify({"error": "unauthorized"}), 401
    try:
        content = request.get_data(as_text=True)
        df = pd.read_csv(io.StringIO(content))
        process_csv(df)  # validate
        _atomic_write_csv(df, LIVE_CSV)  # [W-8] アトミック書き込み
        return jsonify({"status": "ok", "rows": len(df)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def _fl_process():
    """ForexLens 3CSV + YenCast の実勢レートを読み、/forexlens 描画用 JSON を作る。"""
    out = {"forecast": {}, "mood": {}, "cot": {}, "rate": {}}

    # 実勢レートは predictor が毎時 POST する live_data.csv の close を流用する
    # （ForexLens 側に新しい価格配線を持たない）。無ければ空＝パネル非表示。
    try:
        rate_csv = LIVE_CSV if os.path.exists(LIVE_CSV) else SAMPLE_CSV
        r = pd.read_csv(rate_csv)
        r["_t"] = parse_time(r)
        r = r.dropna(subset=["_t"]).sort_values("_t")
        # 地合いチャートと同じ約14日分に絞る
        cutoff = r["_t"].max() - pd.Timedelta(days=14)
        r = r[r["_t"] >= cutoff]
        close = pd.to_numeric(r["close"], errors="coerce")
        out["rate"] = {
            "time":  r["_t"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
            "close": close.round(3).tolist(),
        }
    except Exception:
        pass  # レートが読めなくても ForexLens パネルは表示する

    if os.path.exists(FL_FORECAST_CSV):
        f = pd.read_csv(FL_FORECAST_CSV)
        f["_t"] = pd.to_datetime(f["time_utc"], errors="coerce", utc=True)
        f = f.dropna(subset=["_t"]).sort_values("_t")
        # 表示は JST に統一（YenCast 予測ページと合わせる）
        tj = f["_t"].dt.tz_convert("Asia/Tokyo").dt.strftime("%Y-%m-%dT%H:%M:%S")
        out["forecast"] = {
            "time":   tj.tolist(),
            "score":  pd.to_numeric(f["forecast_score"], errors="coerce").round(4).tolist(),
            "mood":   pd.to_numeric(f.get("mood_score"), errors="coerce").round(4).tolist(),
            "cot":    pd.to_numeric(f.get("cot_score"), errors="coerce").round(4).tolist(),
            "signal": f["signal"].astype(str).tolist(),
        }

    if os.path.exists(FL_MOOD_CSV):
        m = pd.read_csv(FL_MOOD_CSV)
        m["_t"] = pd.to_datetime(m["time_utc"], errors="coerce", utc=True)
        m = m.dropna(subset=["_t"]).sort_values("_t")
        tj = m["_t"].dt.tz_convert("Asia/Tokyo").dt.strftime("%Y-%m-%dT%H:%M:%S")
        out["mood"] = {
            "time": tj.tolist(),
            "bull": pd.to_numeric(m["bull_words"], errors="coerce").tolist(),
            "bear": pd.to_numeric(m["bear_words"], errors="coerce").tolist(),
            "net":  pd.to_numeric(m["net"], errors="coerce").tolist(),
        }

    if os.path.exists(FL_COT_CSV):
        c = pd.read_csv(FL_COT_CSV)
        c["_t"] = pd.to_datetime(c["report_date"], errors="coerce")
        c = c.dropna(subset=["_t"]).sort_values("_t")
        out["cot"] = {
            "date":  c["_t"].dt.strftime("%Y-%m-%d").tolist(),
            "net":   pd.to_numeric(c["net_noncomm"], errors="coerce").tolist(),
            "long":  pd.to_numeric(c["noncomm_long"], errors="coerce").tolist(),
            "short": pd.to_numeric(c["noncomm_short"], errors="coerce").tolist(),
            "oi":    pd.to_numeric(c["open_interest"], errors="coerce").tolist(),
        }

    return _clean_nan(out)


@app.route("/forexlens")
def forexlens():
    try:
        data = _fl_process()
        return render_template("forexlens.html", data=json.dumps(data))
    except Exception as e:
        empty = json.dumps({"forecast": {}, "mood": {}, "cot": {}, "error": str(e)})
        return render_template("forexlens.html", data=empty)


@app.route("/update-forexlens", methods=["POST"])
def update_forexlens():
    # /update と同じトークンで認証（predictor と forex-lens bat が共用）
    if UPDATE_TOKEN:
        auth = request.headers.get("Authorization", "")
        token = auth[7:].strip() if auth.startswith("Bearer ") else ""
        if token != UPDATE_TOKEN:
            return jsonify({"error": "unauthorized"}), 401
    try:
        payload = request.get_json(force=True, silent=False) or {}
        written = {}
        for key, path, required_cols in (
            ("forecast",  FL_FORECAST_CSV, ("time_utc", "forecast_score", "signal")),
            ("news_mood", FL_MOOD_CSV,     ("time_utc", "bull_words", "bear_words", "net")),
            ("cot",       FL_COT_CSV,      ("report_date", "net_noncomm")),
        ):
            text = payload.get(key)
            if not text:
                continue
            df = pd.read_csv(io.StringIO(text))
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                return jsonify({"error": f"{key}: missing columns {missing}"}), 400
            _atomic_write_csv(df, path)
            written[key] = len(df)
        if not written:
            return jsonify({"error": "no known keys in payload"}), 400
        return jsonify({"status": "ok", "written": written})
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
