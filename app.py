import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import requests
import multiprocessing
from typing import List

# --- Optional parallelism (graceful fallback if joblib is not installed) ---
try:
    from joblib import Parallel, delayed  # type: ignore
except Exception:
    Parallel = None
    def delayed(f):
        return f

st.set_page_config(page_title="Rolling Mean Demo", layout="wide")

# -------------------------- UI --------------------------
st.title("互動式 Rolling Mean 視覺化")
st.write("可用隨機資料或 MacroMicro API 進行 rolling mean 計算。")

with st.sidebar:
    st.header("資料來源與參數設定")

    source = st.radio("資料來源", ["Random demo", "MacroMicro API"], horizontal=True)

    # 偵測與視窗參數
    std_choices = [0.5, 1.0, 1.5, 2.0]
    std_list: List[float] = st.multiselect("std（可複選）", options=std_choices, default=[1.0])

    roll_choices = [6, 12, 24, 36, 48]
    winrolling_list: List[int] = st.multiselect("rolling 視窗（月，可複選）", options=roll_choices, default=[12])

    months_gap_threshold: int = st.number_input("事件間隔（至少幾個月）", min_value=1, max_value=24, value=6)

    # Random demo 參數
    if source == "Random demo":
        seed = st.number_input("隨機種子", min_value=0, value=42, step=1)
        n_points = st.slider("資料筆數（月）", min_value=36, max_value=240, value=120, step=12)

    # MacroMicro API 參數
    if source == "MacroMicro API":
        series_ids_text = st.text_input("breath series IDs（逗號分隔）", "10000")
        assetid = st.number_input("index series ID (assetid)", min_value=0, value=0, step=1)
        api_key = st.text_input(
            "MacroMicro API Key（留空則使用 st.secrets）",
            value=st.secrets.get("MACROMICRO_API_KEY", os.environ.get("MACROMICRO_API_KEY", "")),
            type="password"
        )
        st.caption("若已在 .streamlit/secrets.toml 設定 `MACROMICRO_API_KEY`，此欄可留空。")

# ---------------------- Helpers ------------------------
OFFSETS = [-12, -6, 0, 6, 12]  # 以「月」為單位（Row offset on monthly data）

@st.cache_data(show_spinner=False, ttl=3600)
def mm(series_id: int, frequency: str, name: str, api_key: str) -> pd.DataFrame | None:
    """Fetch a series (monthly) from MacroMicro API and return a single-column DataFrame.
    Cached for 1h. Returns None on error.
    """
    url = f"https://dev-biz-api.macromicro.me/v1/stats/series/{series_id}?history=true"
    headers = {"X-Api-Key": api_key}
    for attempt in range(5):
        try:
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()
            df = pd.DataFrame(data["series"]).set_index("date")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index().resample(frequency).mean()
            df.columns = [name]
            return df
        except Exception as e:
            st.write(f"Error fetching series_id {series_id} (attempt {attempt+1}/5): {e}")
            time.sleep(1)
    return None

def nearest_index_pos(idx: pd.DatetimeIndex, ts: pd.Timestamp) -> int:
    try:
        return idx.get_loc(ts)
    except KeyError:
        return idx.get_indexer([ts], method="nearest")[0]

def month_diff(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return (a.year - b.year) * 12 + (a.month - b.month)

def process_series(series_id: int, assetid: int, std_list: List[float], winrolling_list: List[int],
                   api_key: str, months_gap_threshold: int) -> list[dict]:
    results: list[dict] = []

    x1, code1 = "breath", series_id
    x2, code2 = "index", assetid

    df1 = mm(code1, "MS", x1, api_key)
    df2 = mm(code2, "MS", x2, api_key)
    if df1 is None or df2 is None:
        st.warning(f"series_id {series_id} 或 assetid {assetid} 取檔失敗。")
        return results

    alldf_original = pd.concat([df1, df2], axis=1).resample("MS").asfreq().ffill()

    timepast = 12
    timeforward = 12

    for std in std_list:
        for winrolling in winrolling_list:
            try:
                alldf = alldf_original.copy()

                # ===== 第一段分析：原始 breath =====
                df = alldf[[x1, x2]].copy()
                df["Rolling_mean"] = df["breath"].rolling(window=winrolling, min_periods=winrolling).mean()
                df["Rolling_std"] = df["breath"].rolling(window=winrolling, min_periods=winrolling).std()

                cond = (
                    df["breath"].rolling(6, min_periods=6).max()
                    > df["Rolling_mean"] + std * df["Rolling_std"]
                )
                filtered_df = df[cond]

                finalb_dates_1: list[pd.Timestamp] = []
                for dt in filtered_df.index:
                    if not finalb_dates_1 or month_diff(dt, finalb_dates_1[-1]) >= months_gap_threshold:
                        finalb_dates_1.append(dt)

                if not finalb_dates_1:
                    resulttable1 = None
                    finalb1 = None
                    times1 = pre1 = prewin1 = after1 = afterwin1 = score1 = 0
                    effective1 = "no"
                else:
                    dfs = []
                    for dt in finalb_dates_1:
                        idx = nearest_index_pos(alldf.index, pd.Timestamp(dt))
                        if idx - timepast < 0 or idx + timeforward >= len(alldf):
                            continue
                        seg = (
                            alldf["index"].iloc[idx - timepast : idx + timeforward]
                            .to_frame(name=dt)
                            .reset_index()
                        )
                        dfs.append(seg)

                    if not dfs:
                        resulttable1 = None
                        finalb1 = None
                        times1 = pre1 = prewin1 = after1 = afterwin1 = score1 = 0
                        effective1 = "no"
                    else:
                        df_concat = pd.concat(dfs, axis=1)
                        data_cols = [col for j, col in enumerate(df_concat.columns) if j % 2 == 1]
                        origin = df_concat[data_cols]
                        finalb1 = origin.apply(lambda col: 100 * col / col.iloc[timepast])
                        finalb1 = finalb1[finalb1.columns[-10:]]
                        finalb1["median"] = finalb1.mean(axis=1)

                        table1 = pd.concat([finalb1.iloc[timepast + off] for off in OFFSETS], axis=1)
                        table1.columns = [f"{off}m" for off in OFFSETS]
                        resulttable1 = table1.iloc[:-1]
                        # 保持索引為位置即可；如要顯示日期可另外存日期
                        perc_df = pd.DataFrame([(resulttable1 > 100).mean() * 100], index=["勝率"])
                        resulttable1 = pd.concat([resulttable1, perc_df, table1.iloc[-1:]])

                        times1 = len(resulttable1) - 2
                        pre1 = resulttable1.loc["median", "-12m"] - 100
                        prewin1 = resulttable1.loc["勝率", "-12m"]
                        after1 = resulttable1.loc["median", "12m"] - 100
                        afterwin1 = resulttable1.loc["勝率", "12m"]
                        score1 = after1 - pre1
                        effective1 = "yes" if (pre1 - 1) * (after1 - 1) > 0 and times1 > 10 else "no"

                # ===== 第二段分析：breath / breath.shift(12) =====
                df = alldf[[x1, x2]].copy()
                df["breath"] = df["breath"] / df["breath"].shift(12)
                df.dropna(inplace=True)
                df["Rolling_mean"] = df["breath"].rolling(window=winrolling, min_periods=winrolling).mean()
                df["Rolling_std"] = df["breath"].rolling(window=winrolling, min_periods=winrolling).std()
                cond = (
                    df["breath"].rolling(6, min_periods=6).max()
                    > df["Rolling_mean"] + std * df["Rolling_std"]
                )
                filtered_df = df[cond]

                finalb_dates_2: list[pd.Timestamp] = []
                for dt in filtered_df.index:
                    if not finalb_dates_2 or month_diff(dt, finalb_dates_2[-1]) >= months_gap_threshold:
                        finalb_dates_2.append(dt)

                if not finalb_dates_2:
                    resulttable2 = None
                    finalb2 = None
                    times2 = pre2 = prewin2 = after2 = afterwin2 = score2 = 0
                    effective2 = "no"
                else:
                    dfs = []
                    for dt in finalb_dates_2:
                        idx = nearest_index_pos(alldf.index, pd.Timestamp(dt))
                        if idx - timepast < 0 or idx + timeforward >= len(alldf):
                            continue
                        seg = (
                            alldf["index"].iloc[idx - timepast : idx + timeforward]
                            .to_frame(name=dt)
                            .reset_index()
                        )
                        dfs.append(seg)

                    if not dfs:
                        resulttable2 = None
                        finalb2 = None
                        times2 = pre2 = prewin2 = after2 = afterwin2 = score2 = 0
                        effective2 = "no"
                    else:
                        df_concat = pd.concat(dfs, axis=1)
                        data_cols = [col for j, col in enumerate(df_concat.columns) if j % 2 == 1]
                        origin = df_concat[data_cols]
                        finalb2 = origin.apply(lambda col: 100 * col / col.iloc[timepast])
                        finalb2["median"] = finalb2.mean(axis=1)

                        table2 = pd.concat([finalb2.iloc[timepast + off] for off in OFFSETS], axis=1)
                        table2.columns = [f"{off}m" for off in OFFSETS]
                        resulttable2 = table2.iloc[:-1]
                        perc_df = pd.DataFrame([(resulttable2 > 100).mean() * 100], index=["勝率"])
                        resulttable2 = pd.concat([resulttable2, perc_df, table2.iloc[-1:]])

                        times2 = len(resulttable2) - 2
                        pre2 = resulttable2.loc["median", "-12m"] - 100
                        prewin2 = resulttable2.loc["勝率", "-12m"]
                        after2 = resulttable2.loc["median", "12m"] - 100
                        afterwin2 = resulttable2.loc["勝率", "12m"]
                        score2 = after2 - pre2
                        effective2 = "yes" if (pre2 - 1) * (after2 - 1) > 0 and times2 > 10 else "no"

                results.append({
                    "series_id": series_id,
                    "std": std,
                    "winrolling": winrolling,
                    "pre1": pre1, "prewin1": prewin1, "after1": after1, "afterwin1": afterwin1,
                    "times1": times1, "effective1": effective1,
                    "pre2": pre2, "prewin2": prewin2, "after2": after2, "afterwin2": afterwin2,
                    "times2": times2, "effective2": effective2,
                    "resulttable1": resulttable1 if resulttable1 is not None else None,
                    "finalb1": finalb1.reset_index() if finalb1 is not None else None,
                    "resulttable2": resulttable2 if resulttable2 is not None else None,
                    "finalb2": finalb2.reset_index() if finalb2 is not None else None,
                })
            except Exception as e:
                st.error(f"計算錯誤（series {series_id}）：{e}")

    return results

# ---------------------- Main Flow ----------------------
if source == "Random demo":
    rng = np.random.default_rng(int(seed))
    vals = rng.integers(10, 1000, size=int(n_points)).astype(float)
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=int(n_points), freq="MS")
    df_demo = pd.DataFrame({"value": vals}, index=idx)

    st.subheader("Random Demo 資料預覽")
    st.dataframe(df_demo.tail())

    for win in winrolling_list:
        if win <= 1:
            continue
        st.write(f"Rolling mean（{win} 個月）")
        st.line_chart(df_demo[["value"]].rolling(win).mean())

else:  # MacroMicro API
    effective_api_key = st.secrets.get("MACROMICRO_API_KEY", "") or api_key or os.environ.get("MACROMICRO_API_KEY", "")
    if not effective_api_key:
        st.error("缺少 MacroMicro API Key。請在 .streamlit/secrets.toml 設定或於側邊欄輸入。")
        st.stop()

    try:
        series_ids = [int(s.strip()) for s in series_ids_text.split(",") if s.strip()]
    except Exception:
        st.error("Series IDs 格式錯誤。請以逗號分隔整數 ID。")
        st.stop()

    # 平行或單執行緒
    if Parallel is not None:
        num_cores = max(1, min(4, multiprocessing.cpu_count()))
        results_nested = Parallel(n_jobs=num_cores)(
            delayed(process_series)(sid, assetid, std_list, winrolling_list, effective_api_key, months_gap_threshold)
            for sid in series_ids
        )
        results_flat = [item for sublist in results_nested for item in sublist]
    else:
        st.warning("`joblib` 未安裝，改用單執行緒。")
        results_flat = []
        for sid in series_ids:
            results_flat.extend(
                process_series(sid, assetid, std_list, winrolling_list, effective_api_key, months_gap_threshold)
            )

    if not results_flat:
        st.info("尚無可顯示結果。請調整參數或確認 series 有足夠歷史資料。")
        st.stop()

    summary_df = pd.DataFrame([
        {k: v for k, v in r.items() if 'resulttable' not in k and 'finalb' not in k}
        for r in results_flat
    ])

    st.subheader("匯總結果（Summary）")
    st.dataframe(summary_df)

    # 範例表格
    resulttable1_list = [pd.DataFrame(r['resulttable1']) for r in results_flat if r.get('resulttable1') is not None]
    resulttable2_list = [pd.DataFrame(r['resulttable2']) for r in results_flat if r.get('resulttable2') is not None]
    finalb1_list = [pd.DataFrame(r['finalb1']) for r in results_flat if r.get('finalb1') is not None]
    finalb2_list = [pd.DataFrame(r['finalb2']) for r in results_flat if r.get('finalb2') is not None]

    if resulttable1_list:
        st.write("=== resulttable1 範例 ===")
        st.dataframe(resulttable1_list[0].head())

    if resulttable2_list:
        st.write("=== resulttable2 範例 ===")
        st.dataframe(resulttable2_list[0].head())

    if finalb1_list:
        st.write("=== finalb1 範例 ===")
        st.dataframe(finalb1_list[0].head())

    if finalb2_list:
        st.write("=== finalb2 範例 ===")
        st.dataframe(finalb2_list[0].head())
