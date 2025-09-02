import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt
from datetime import datetime
from joblib import Parallel, delayed
import multiprocessing
import plotly.graph_objects as go
import altair as alt
alt.data_transformers.disable_max_rows()

# ===== 內建 (GitHub) 對照表載入：不透過上傳 =====
MAP_PATH = os.getenv("ID_NAME_MAP_PATH", "data/id_name_map.xlsx")  # 預設讀取 repo 內的檔案

@st.cache_data(show_spinner=False)
def load_mapping_from_repo(path: str):
    """從 repo 內建檔案載入 ID→中文名對照表，支援 CSV / Excel。
    回傳 (series_name_map, asset_name_map, df_preview)。找不到/失敗則回傳空字典。"""
    import io
    try:
        with open(path, "rb") as f:
            file_bytes = f.read()
        ext = os.path.splitext(path)[1]
        series_map, asset_map, df_preview = _parse_mapping(file_bytes, ext)
        return series_map, asset_map, df_preview
    except Exception as e:
        return {}, {}, pd.DataFrame()

# 全域名稱對照：供整個 App 使用
series_name_map, asset_name_map, df_preview = load_mapping_from_repo(MAP_PATH)


# --- Optional parallelism ---
try:
    from joblib import Parallel, delayed
except Exception:
    Parallel = None
    def delayed(f):
        return f

st.set_page_config(page_title="熊市訊號與牛市訊號尋找工具", layout="wide")

# -------------------------- Helpers --------------------------
OFFSETS = [-12, -6, 0, 6, 12]  # 以「月」為單位
sigma_levels = [0.5, 1.0, 1.5, 2.0]

@st.cache_data(show_spinner=False)
def _parse_mapping(file_bytes: bytes, ext: str):
    """讀取使用者上傳的ID→中文名稱對照表（支援 CSV / XLSX）。
    回傳 (series_name_map, asset_name_map, df_preview)
    可辨識的欄名（不分大小寫）：
      變數ID: ["series_id", "變數id", "變數ID", "Series ID"]
      變數名稱: ["series_name", "變數名稱", "變數中文名稱", "Series Name"]
      研究目標ID: ["asset_id", "研究目標id", "研究目標ID", "資產ID"]
      研究目標名稱: ["asset_name", "研究目標名稱", "研究目標中文名稱", "Asset Name"]
    """
    import io
    if ext.lower() in (".csv",):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))

    def match_col(options: list[str]) -> str | None:
        # 先精確比對
        for opt in options:
            if opt in df.columns:
                return opt
        # 再用不分大小寫比對
        low = {c.lower(): c for c in df.columns}
        for opt in options:
            if opt.lower() in low:
                return low[opt.lower()]
        return None

    s_id_col   = match_col(["series_id", "變數id", "變數ID", "Series ID"]) 
    s_name_col = match_col(["series_name", "變數名稱", "變數中文名稱", "Series Name"]) 
    a_id_col   = match_col(["asset_id", "研究目標id", "研究目標ID", "資產ID"]) 
    a_name_col = match_col(["asset_name", "研究目標名稱", "研究目標中文名稱", "Asset Name"]) 

    series_map: dict[int, str] = {}
    asset_map: dict[int, str] = {}

    if s_id_col and s_name_col:
        try:
            series_map = {int(k): str(v) for k, v in zip(df[s_id_col], df[s_name_col]) if pd.notna(k) and pd.notna(v)}
        except Exception:
            pass
    if a_id_col and a_name_col:
        try:
            asset_map = {int(k): str(v) for k, v in zip(df[a_id_col], df[a_name_col]) if pd.notna(k) and pd.notna(v)}
        except Exception:
            pass

    # 預覽欄位（不改原始 df）
    preview_cols = [c for c in [s_id_col, s_name_col, a_id_col, a_name_col] if c]
    df_preview = df[preview_cols].head(10) if preview_cols else df.head(5)
    return series_map, asset_map, df_preview


def _need_api_key() -> str:
    k = (
        st.session_state.get("__api_key__")
        or st.secrets.get("MACROMICRO_API_KEY", "")
        or os.environ.get("MACROMICRO_API_KEY", "")
    )
    if not k:
        st.error("缺少 MacroMicro API Key。請在側邊欄輸入或於 .streamlit/secrets.toml 設定。")
        st.stop()
    return k

@st.cache_data(show_spinner=False, ttl=3600)
def mm(series_id: int, frequency: str, name: str, k: str) -> pd.DataFrame | None:
    """抓單一序列（月頻），回傳單欄 DataFrame；錯誤回 None。"""
    url = f"https://dev-biz-api.macromicro.me/v1/stats/series/{series_id}?history=true"
    headers = {"X-Api-Key": k}
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


def find_row_number_for_date(df_obj: pd.DataFrame, specific_date: pd.Timestamp) -> int:
    return df_obj.index.get_loc(pd.Timestamp(specific_date))


def _condition(df: pd.DataFrame, std: float, winrolling: int, mode: str) -> pd.Series:
    """
    根據選擇的觸發邏輯回傳布林條件：
    - Greater：過去6月最高 > 均值 + std*標準差
    - Smaller：過去6月最低 < 均值 - std*標準差
    """
    if mode == "Greater":
        return df["breath"].rolling(6).max() > df["Rolling_mean"] + std * df["Rolling_std"]
    else:
        return df["breath"].rolling(6).min() < df["Rolling_mean"] - std * df["Rolling_std"]


def resolve_name(id_int: int, name_map: dict[int, str]) -> str:
    return name_map.get(int(id_int), str(int(id_int)))


# 主分析（保留原流程，新增名稱資訊在結果中）
def process_series(series_id: int, std_value: float, winrolling_value: int, k: str, mode: str,
                   series_name_map: dict[int, str], asset_name_map: dict[int, str], assetid: int) -> list[dict]:
    results: list[dict] = []
    try:
        x1, code1 = "breath", series_id
        x2, code2 = "index", assetid

        # 解析中文名稱（僅用於顯示，不影響運算欄位名）
        series_label = resolve_name(code1, series_name_map)
        asset_label  = resolve_name(code2, asset_name_map)

        df1 = mm(code1, "MS", x1, k)
        df2 = mm(code2, "MS", x2, k)
        if df1 is None or df2 is None:
            st.warning(f"series_id {series_id} 或 assetid {assetid} 取檔失敗。")
            return results

        alldf_original = pd.concat([df1, df2], axis=1).resample("MS").asfreq().ffill()

        alldf = alldf_original.copy()
        timeforward, timepast = 31, 31  # 定義 timepast 和 timeforward
        months_threshold = st.session_state.get("months_gap_threshold", 6)

        # ===== 第一段分析：原始 breath =====
        df = alldf[[x1, x2]].copy()
        df["Rolling_mean"] = df["breath"].rolling(window=winrolling_value).mean()
        df["Rolling_std"] = df["breath"].rolling(window=winrolling_value).std()
        filtered_df = df[_condition(df, std_value, winrolling_value, mode)]

        finalb_dates_1: list[pd.Timestamp] = []
        for date in filtered_df.index:
            if not finalb_dates_1 or ((date - finalb_dates_1[-1]).days / 30) >= months_threshold:
                finalb_dates_1.append(date)

        if not finalb_dates_1:
            resulttable1 = None
            finalb1 = None
            times1 = pre1 = prewin1 = after1 = afterwin1 = score1 = 0
            effective1 = "no"
        else:
            dfs = []
            for dt in finalb_dates_1:
                a = find_row_number_for_date(alldf, dt)
                if a - timepast < 0 or a + timeforward >= len(alldf):
                    continue
                temp_df = alldf["index"].iloc[a - timepast : a + timeforward].to_frame(name=dt).reset_index()
                dfs.append(temp_df)
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
                finalb1 = finalb1[finalb1.columns[-10:]]  # 只保留最近 10 次事件
                finalb1["median"] = finalb1.mean(axis=1)

                offsets = [-12, -6, 0, 6, 12]
                table1 = pd.concat([finalb1.iloc[timepast + off] for off in offsets], axis=1)
                table1.columns = [f"{off}d" for off in offsets]  # 沿用 d 命名
                resulttable1 = table1.iloc[:-1]
                perc_df = pd.DataFrame([(resulttable1 > 100).mean() * 100], index=["勝率"])
                resulttable1 = pd.concat([resulttable1, perc_df, table1.iloc[-1:]])

                times1 = len(resulttable1) - 2
                pre1 = resulttable1.loc["median", "-12d"] - 100
                prewin1 = resulttable1.loc["勝率", "-12d"]
                after1 = resulttable1.loc["median", "12d"] - 100
                afterwin1 = resulttable1.loc["勝率", "12d"]
                score1 = after1 - pre1
                effective1 = "yes" if (pre1 - 1) * (after1 - 1) > 0 and times1 > 10 else "no"

        # ===== 第二段分析：breath / breath.shift(12) =====
        df = alldf[[x1, x2]].copy()
        df["breath"] = df["breath"] / df["breath"].shift(12)
        df.dropna(inplace=True)
        df["Rolling_mean"] = df["breath"].rolling(window=winrolling_value).mean()
        df["Rolling_std"] = df["breath"].rolling(window=winrolling_value).std()
        filtered_df = df[_condition(df, std_value, winrolling_value, mode)]

        finalb_dates_2: list[pd.Timestamp] = []
        for date in filtered_df.index:
            if not finalb_dates_2 or ((date - finalb_dates_2[-1]).days / 30) >= months_threshold:
                finalb_dates_2.append(date)

        if not finalb_dates_2:
            resulttable2 = None
            finalb2 = None
            times2 = pre2 = prewin2 = after2 = afterwin2 = score2 = 0
            effective2 = "no"
        else:
            dfs = []
            for dt in finalb_dates_2:
                a = find_row_number_for_date(alldf, dt)
                if a - timepast < 0 or a + timeforward >= len(alldf):
                    continue
                temp_df = alldf["index"].iloc[a - timepast : a + timeforward].to_frame(name=dt).reset_index()
                dfs.append(temp_df)
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

                offsets = [-12, -6, 0, 6, 12]
                table2 = pd.concat([finalb2.iloc[timepast + off] for off in offsets], axis=1)
                table2.columns = [f"{off}d" for off in offsets]
                resulttable2 = table2.iloc[:-1]
                perc_df = pd.DataFrame([(resulttable2 > 100).mean() * 100], index=["勝率"])
                resulttable2 = pd.concat([resulttable2, perc_df, table2.iloc[-1:]])

                times2 = len(resulttable2) - 2
                pre2 = resulttable2.loc["median", "-12d"] - 100
                prewin2 = resulttable2.loc["勝率", "-12d"]
                after2 = resulttable2.loc["median", "12d"] - 100
                afterwin2 = resulttable2.loc["勝率", "12d"]
                score2 = after2 - pre2
                effective2 = "yes" if (pre2 - 1) * (after2 - 1) > 0 and times2 > 10 else "no"

        results.append({
            "series_id": series_id,
            "series_label": series_label,
            "asset_label": asset_label,
            "std": std_value,
            "winrolling": winrolling_value,
            "pre1": pre1, "prewin1": prewin1, "after1": after1, "afterwin1": afterwin1,
            "times1": times1, "effective1": effective1,
            "pre2": pre2, "prewin2": prewin2, "after2": after2, "afterwin2": afterwin2,
            "times2": times2, "effective2": effective2,
            "resulttable1": resulttable1 if resulttable1 is not None else None,
            "resulttable2": resulttable2 if resulttable2 is not None else None,
            "finalb1": finalb1.reset_index() if finalb1 is not None else None,
            "finalb2": finalb2.reset_index() if finalb2 is not None else None,
        })

    except Exception as e:
        st.write(f"Error during CALCULATION for series {series_id}: {e}")
    return results


# -------------------------- UI --------------------------
st.title("熊市訊號與牛市訊號尋找工具")

with st.sidebar:
    st.header("資料來源與參數設定")

    # 讀取 repo 內建對照表（不需上傳）
if series_name_map or asset_name_map:
    st.success(f"已載入對照表：變數 {len(series_name_map)} 筆、研究目標 {len(asset_name_map)} 筆。來源：{MAP_PATH}")
    with st.expander("對照表預覽（前10列）", expanded=False):
        st.dataframe(df_preview if not df_preview.empty else pd.DataFrame({"提示": ["對照表內容為空"]}))
else:
    st.info(f"找不到對照表檔案：{MAP_PATH}，將以數字ID顯示。可透過環境變數 ID_NAME_MAP_PATH 指定路徑。")
# 觸發邏輯選擇：Greater / Smaller
    trigger_mode = st.radio("觸發邏輯", ["Greater", "Smaller"], horizontal=True)

    series_ids_text = st.text_input("變數ID（逗號分隔）", "10000")

    assetid = st.number_input("研究目標ID", min_value=0, value=0, step=1)
    # 顯示對應名稱（若未提供對照表或查無，則顯示原始數字）
    try:
        if series_ids_text.strip():
            _sids = [int(s.strip()) for s in series_ids_text.split(",") if s.strip()]
            series_labels = [series_name_map.get(s, str(s)) for s in _sids]
            st.caption("變數：" + "、".join(series_labels))
    except Exception:
        pass
    try:
        st.caption("研究目標：" + asset_name_map.get(int(assetid), str(int(assetid))))
    except Exception:
        pass

    api_key = st.text_input(
        "MacroMicro API Key（留空則使用 st.secrets 或環境變數）",
        value=st.secrets.get("MACROMICRO_API_KEY", os.environ.get("MACROMICRO_API_KEY", "")),
        type="password"
    )
    # 保存至 session_state 供 _need_api_key() 取用
    st.session_state["__api_key__"] = api_key

    # 單選的 std 和 rolling 視窗（這些值在兩段分析中共用）
    std_choices = [0.5, 1.0, 1.5, 2.0]
    std_value = st.selectbox("標準差門檻", options=std_choices, index=1)

    roll_choices = [6, 12, 24, 36, 60, 120]
    winrolling_value = st.selectbox("滾動期數", options=roll_choices, index=1)

    months_gap_threshold = st.number_input("事件間隔（至少幾個月）", min_value=1, max_value=36, value=6)
    st.session_state["months_gap_threshold"] = months_gap_threshold

# ---------------------- Main Flow ----------------------
try:
    series_ids = [int(s.strip()) for s in series_ids_text.split(",") if s.strip()]
except Exception:
    st.error("Series IDs 格式錯誤。請以逗號分隔整數 ID。")
    st.stop()

std_value = std_value
winrolling_value = winrolling_value
mode = trigger_mode
k = _need_api_key()

# 平行執行（或退回單執行緒）
if Parallel is not None:
    num_cores = max(1, min(4, multiprocessing.cpu_count()))
    results_nested = Parallel(n_jobs=num_cores)(
        delayed(process_series)(sid, std_value, winrolling_value, k, mode, series_name_map, asset_name_map, int(assetid))
        for sid in series_ids
    )
    results_flat = [item for sublist in results_nested for item in sublist]
else:
    st.warning("`joblib` 未安裝，改用單執行緒。")
    results_flat = []
    for sid in series_ids:
        results_flat.extend(process_series(sid, std_value, winrolling_value, k, mode, series_name_map, asset_name_map, int(assetid)))

if not results_flat:
    st.info("尚無可顯示結果。請調整參數或確認 series 有足夠歷史資料。")
    st.stop()

# ===== 第一段分析：原始 breath =====
st.subheader("原始值版本")
# 顯示選取的名稱（以第一個結果為準）
try:
    st.caption(f"變數：{results_flat[0]['series_label']} ｜ 研究目標：{results_flat[0]['asset_label']}")
except Exception:
    pass

resulttable1_list = [r['resulttable1'] for r in results_flat if r.get('resulttable1') is not None]
if not resulttable1_list:
    st.warning("第一段分析尚無有效表格結果。")
else:
    df = resulttable1_list[0]
    WIN_RATE_LABEL = "勝率"
    pre    = float(df.loc['median', '-12d']) - 100
    after  = float(df.loc['median', '12d'])  - 100
    prewin = float(df.loc[WIN_RATE_LABEL, '-12d'])
    afterwin = float(df.loc[WIN_RATE_LABEL, '12d'])
    times = len(df) - 2
    effectivepart1 = (
        '為有效訊號'
        if ((pre - 1) * (after - 1) > 0) and (times > 10) and ((prewin + afterwin > 140) or (prewin + afterwin < 60))
        else '不是有效訊號'
    )
    st.subheader(effectivepart1)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(df, use_container_width=True)
    with col2:
        first_finalb1 = next((r['finalb1'] for r in results_flat if r.get('finalb1') is not None), None)
        if first_finalb1 is not None:
            x = np.linspace(-31, 31, 31 + 31)
            y = first_finalb1['median']
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(x, y, label='事件中位數路徑')
            ax.axvline(0, color='grey', linestyle='--')
            xlim = (-15, 15)
            ax.set_xlim(xlim)
            ax.set_ylim(
                bottom=y[(x >= xlim[0]) & (x <= xlim[1])].min() * 0.99,
                top=y[(x >= xlim[0]) & (x <= xlim[1])].max() * 1.01)
            ax.set_xlabel('事件相對月份')
            ax.set_ylabel('指數（=100 為事件月）')
            ax.legend()
            st.pyplot(fig, use_container_width=True)

st.divider()

# ===== 第二段分析：breath / breath.shift(12) =====
st.subheader("年增率版本")
try:
    st.caption(f"變數：{results_flat[0]['series_label']} ｜ 研究目標：{results_flat[0]['asset_label']}")
except Exception:
    pass

resulttable2_list = [r['resulttable2'] for r in results_flat if r.get('resulttable2') is not None]
if not resulttable2_list:
    st.warning("第二段分析尚無有效表格結果。")
else:
    df2 = resulttable2_list[0]
    WIN_RATE_LABEL = "勝率"
    pre    = float(df2.loc['median', '-12d']) - 100
    after  = float(df2.loc['median', '12d'])  - 100
    prewin = float(df2.loc[WIN_RATE_LABEL, '-12d'])
    afterwin = float(df2.loc[WIN_RATE_LABEL, '12d'])
    times = len(df2) - 2
    effectivepart2 = (
        '為有效訊號'
        if ((pre - 1) * (after - 1) > 0) and (times > 10) and ((prewin + afterwin > 140) or (prewin + afterwin < 60))
        else '不是有效訊號'
    )
    st.subheader(effectivepart2)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(df2, use_container_width=True)
    with col2:
        first_finalb2 = next((r['finalb2'] for r in results_flat if r.get('finalb2') is not None), None)
        if first_finalb2 is not None:
            x = np.linspace(-31, 31, 31 + 31)
            y = first_finalb2['median']
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(x, y, label='事件中位數路徑')
            ax.axvline(0, color='grey', linestyle='--')
            xlim = (-15, 15)
            ax.set_xlim(xlim)
            ax.set_ylim(
                bottom=y[(x >= xlim[0]) & (x <= xlim[1])].min() * 0.99,
                top=y[(x >= xlim[0]) & (x <= xlim[1])].max() * 1.01)
            ax.set_xlabel('事件相對月份')
            ax.set_ylabel('指數（=100 為事件月）')
            ax.legend()
            st.pyplot(fig, use_container_width=True)

# ===== Plot by series_ids_text: Levels & YoY (brush to set x-range; y auto-rescales) =====
st.divider()
st.subheader("Each 變數：原始值（rolling mean ±σ）與年增率（可框選時間區間）")


def levels_chart_with_brush(s: pd.Series, sid: int, label: str):
    roll_mean = s.rolling(winrolling_value).mean()
    roll_std  = s.rolling(winrolling_value).std()

    df_levels = pd.DataFrame({
        "Date": s.index,
        "Level": s.values,
        "Mean": roll_mean.values,
    })
    # add ±σ bands
    for m in sigma_levels:
        df_levels[f"+{m}σ"] = (roll_mean + m * roll_std).values
        df_levels[f"-{m}σ"] = (roll_mean - m * roll_std).values

    long_levels = df_levels.melt("Date", var_name="Series", value_name="Value").dropna()
    brush = alt.selection_interval(encodings=["x"])

    upper = (
        alt.Chart(long_levels)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Value:Q", title="Level"),
            color=alt.Color("Series:N", legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("Date:T"), "Series:N", alt.Tooltip("Value:Q", format=".2f")],
        )
        .transform_filter(brush)
        .properties(title=f"{label}（ID: {sid}） | {winrolling_value}-period rolling mean ±σ", height=320)
    )

    lower = (
        alt.Chart(df_levels)
        .mark_area(opacity=0.4)
        .encode(x=alt.X("Date:T", title=""), y=alt.Y("Level:Q", title=""))
        .properties(height=60)
        .add_selection(brush)
    )

    return alt.vconcat(upper, lower).resolve_scale(y="independent")


def yoy_chart_with_brush(s: pd.Series, sid: int, label: str):
    yoy = s.pct_change(12) * 100.0
    yoy_mean = yoy.rolling(winrolling_value).mean()
    yoy_std  = yoy.rolling(winrolling_value).std()

    df_yoy = pd.DataFrame({
        "Date": yoy.index,
        "YoY (%)": yoy.values,
        "Mean": yoy_mean.values,
    })
    for m in sigma_levels:
        df_yoy[f"+{m}σ"] = (yoy_mean + m * yoy_std).values
        df_yoy[f"-{m}σ"] = (yoy_mean - m * yoy_std).values

    long_yoy = df_yoy.melt("Date", var_name="Series", value_name="Value").dropna()

    brush = alt.selection_interval(encodings=["x"])

    upper = (
        alt.Chart(long_yoy)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Value:Q", title="YoY (%)"),
            color=alt.Color("Series:N", legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("Date:T"), "Series:N", alt.Tooltip("Value:Q", format=".2f")],
        )
        .transform_filter(brush)
        .properties(title=f"{label}（ID: {sid}） | YoY (%) with {winrolling_value}-period rolling mean ±σ", height=320)
    )

    zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4,4]).encode(y="y:Q")

    lower = (
        alt.Chart(df_yoy)
        .mark_area(opacity=0.4)
        .encode(x=alt.X("Date:T", title=""), y=alt.Y("YoY (%):Q", title=""))
        .properties(height=60)
        .add_selection(brush)
    )

    return alt.vconcat(upper + zero_line, lower).resolve_scale(y="independent")

for sid in series_ids:
    df_target = mm(int(sid), "MS", f"series_{sid}", k)
    if df_target is None or df_target.empty:
        st.info(f"No data for series {sid}, skipping.")
        continue

    s = df_target.iloc[:, 0].astype(float)
    label = series_name_map.get(int(sid), str(int(sid)))

    with st.expander(f"{label}（ID: {sid}）", expanded=False):
        colA, colB = st.columns(2)
        with colA:
            st.altair_chart(levels_chart_with_brush(s, sid, label), use_container_width=True)
        with colB:
            st.altair_chart(yoy_chart_with_brush(s, sid, label), use_container_width=True)
