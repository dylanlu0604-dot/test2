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
import io

# --- Optional parallelism ---
try:
    from joblib import Parallel, delayed
except Exception:
    Parallel = None
    def delayed(f):
        return f

st.set_page_config(page_title="熊市訊號與牛市訊號尋找工具", layout="wide")

# -------------------------- UI --------------------------
st.title("熊市訊號與牛市訊號尋找工具")

# --- Function to load ID to Name mapping from GitHub ---
@st.cache_data(show_spinner="下載ID對應表...", ttl=3600)
def load_series_id_map() -> pd.DataFrame:
    """從 GitHub 下載 ID 與名稱對應的 Excel 檔案。"""
    github_url = "https://raw.githubusercontent.com/dylanlu0604-dot/test2/main/Idwithname.xlsx"
    try:
        response = requests.get(github_url)
        response.raise_for_status() # 檢查是否有 HTTP 錯誤
        df = pd.read_excel(io.BytesIO(response.content))
        
        # Check for expected columns and handle potential errors
        required_cols = ['ID', '名稱']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Excel 檔案中缺少必要的欄位。預期欄位：{required_cols}。實際欄位：{df.columns.tolist()}")
            st.stop()
        
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"無法從 GitHub 下載對應表: {e}")
        st.stop()
    except Exception as e:
        st.error(f"處理 Excel 檔案時發生錯誤: {e}")
        st.stop()

# 載入 ID 對應表
id_name_map = load_series_id_map()
# 處理 NaN 或空值
id_name_map = id_name_map.dropna(subset=['ID', '名稱']).astype({'ID': int, '名稱': str})
series_names = id_name_map['名稱'].tolist()


with st.sidebar:
    st.header("資料來源與參數設定")
    
    # 觸發邏輯選擇：Greater / Smaller
    trigger_mode = st.radio("觸發邏輯", ["Greater", "Smaller"], horizontal=True)

    # 將文字輸入框替換為下拉式選單，顯示中文名稱
    selected_name = st.selectbox("變數ID", options=series_names, index=0)
    # 根據選定的中文名稱找出對應的 ID
    selected_id = id_name_map[id_name_map['名稱'] == selected_name]['ID'].iloc[0]

    assetid = st.number_input("研究目標ID", min_value=0, value=0, step=1)
    api_key = st.text_input(
        "MacroMicro API Key（留空則使用 st.secrets 或環境變數）",
        value=st.secrets.get("MACROMICRO_API_KEY", os.environ.get("MACROMICRO_API_KEY", "")),
        type="password"
    )

    # 單選的 std 和 rolling 視窗（這些值在兩段分析中共用）
    std_choices = [0.5, 1.0, 1.5, 2.0]
    std_value = st.selectbox("標準差門檻", options=std_choices, index=1)

    roll_choices = [6, 12, 24, 36, 60,120]
    winrolling_value = st.selectbox("滾動期數", options=roll_choices, index=1)

    months_gap_threshold = st.number_input("事件間隔（至少幾個月）", min_value=1, max_value=36, value=6)

# ---------------------- Helpers ------------------------
OFFSETS = [-12, -6, 0, 6, 12]  # 以「月」為單位

def _need_api_key() -> str:
    k = api_key or st.secrets.get("MACROMICRO_API_KEY", "") or os.environ.get("MACROMICRO_API_KEY", "")
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

# 主分析（保留你的原流程，只把條件改為可切換）
def process_series(series_id: int, std_value: float, winrolling_value: int, k: str, mode: str) -> list[dict]:
    results: list[dict] = []
    try:
        x1, code1 = "breath", series_id
        x2, code2 = "index", assetid

        df1 = mm(code1, "MS", x1, k)
        df2 = mm(code2, "MS", x2, k)
        if df1 is None or df2 is None:
            st.warning(f"series_id {series_id} 或 assetid {assetid} 取檔失敗。")
            return results

        alldf_original = pd.concat([df1, df2], axis=1).resample("MS").asfreq().ffill()

        alldf = alldf_original.copy()
        timeforward, timepast = 31, 31  # 定義 timepast 和 timeforward
        months_threshold = months_gap_threshold

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
                finalb1["mean"] = finalb1.mean(axis=1)

                offsets = [-12, -6, 0, 6, 12]
                table1 = pd.concat([finalb1.iloc[timepast + off] for off in offsets], axis=1)
                table1.columns = [f"{off}d" for off in offsets]  # 仍沿用 d 命名
                resulttable1 = table1.iloc[:-1]
                perc_df = pd.DataFrame([(resulttable1 > 100).mean() * 100], index=["勝率"])
                resulttable1 = pd.concat([resulttable1, perc_df, table1.iloc[-1:]])

                times1 = len(resulttable1) - 2
                pre1 = resulttable1.loc["mean", "-12d"] - 100 if "mean" in resulttable1.index else 0
                prewin1 = resulttable1.loc["勝率", "-12d"] if "勝率" in resulttable1.index else 0
                after1 = resulttable1.loc["mean", "12d"] - 100 if "mean" in resulttable1.index else 0
                afterwin1 = resulttable1.loc["勝率", "12d"] if "勝率" in resulttable1.index else 0
                score1 = after1 - pre1
                effective1 = "yes" if (pre1 > 0 and after1 > 0) or (pre1 < 0 and after1 < 0) and times1 > 10 else "no"

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
                finalb2["mean"] = finalb2.mean(axis=1)

                offsets = [-12, -6, 0, 6, 12]
                table2 = pd.concat([finalb2.iloc[timepast + off] for off in offsets], axis=1)
                table2.columns = [f"{off}d" for off in offsets]
                resulttable2 = table2.iloc[:-1]
                perc_df = pd.DataFrame([(resulttable2 > 100).mean() * 100], index=["勝率"])
                resulttable2 = pd.concat([resulttable2, perc_df, table2.iloc[-1:]])

                times2 = len(resulttable2) - 2
                pre2 = resulttable2.loc["mean", "-12d"] - 100 if "mean" in resulttable2.index else 0
                prewin2 = resulttable2.loc["勝率", "-12d"] if "勝率" in resulttable2.index else 0
                after2 = resulttable2.loc["mean", "12d"] - 100 if "mean" in resulttable2.index else 0
                afterwin2 = resulttable2.loc["勝率", "12d"] if "勝率" in resulttable2.index else 0
                score2 = after2 - pre2
                effective2 = "yes" if (pre2 > 0 and after2 > 0) or (pre2 < 0 and after2 < 0) and times2 > 10 else "no"

        results.append({
            "series_id": series_id, "std": std_value, "winrolling": winrolling_value,
            "pre1": pre1, "prewin1": prewin1, "after1": after1, "afterwin1": afterwin1,
            "times1": times1, "effective1": effective1,
            "pre2": pre2, "prewin2": prewin2, "after2": after2, "afterwin2": afterwin2,
            "times2": times2, "effective2": effective2,
            "resulttable1": resulttable1 if resulttable1 is not None else None,
            "resulttable2": resulttable2 if resulttable2 is not None else None,
            "finalb1": finalb1.reset_index() if finalb1 is not None and "mean" in finalb1.columns else None,
            "finalb2": finalb2.reset_index() if finalb2 is not None and "mean" in finalb2.columns else None,
        })

    except Exception as e:
        st.write(f"Error during CALCULATION for series {series_id}: {e}")
    return results


# ---------------------- Main Flow ----------------------

series_ids = [selected_id] # 取得下拉式選單的 ID
std_value = std_value
winrolling_value = winrolling_value
mode = trigger_mode
k = _need_api_key()

# 平行執行（或退回單執行緒）
if Parallel is not None:
    num_cores = max(1, min(4, multiprocessing.cpu_count()))
    results_nested = Parallel(n_jobs=num_cores)(
        delayed(process_series)(sid, std_value, winrolling_value, k, mode) for sid in series_ids
    )
    results_flat = [item for sublist in results_nested for item in sublist]
else:
    st.warning("`joblib` 未安裝，改用單執行緒。")
    results_flat = []
    for sid in series_ids:
        results_flat.extend(process_series(sid, std_value, winrolling_value, k, mode))

if not results_flat:
    st.info("尚無可顯示結果。請調整參數或確認 series 有足夠歷史資料。")
    st.stop()


# ===== 第一段分析：原始 breath =====
st.subheader("原始值版本")

resulttable1_list = [r['resulttable1'] for r in results_flat if r.get('resulttable1') is not None]

if resulttable1_list:
    df = resulttable1_list[0]
    WIN_RATE_LABEL = "勝率"

    pre = float(df.loc['mean', '-12d']) - 100 if 'mean' in df.index and '-12d' in df.columns else 0
    after = float(df.loc['mean', '12d']) - 100 if 'mean' in df.index and '12d' in df.columns else 0
    prewin = float(df.loc[WIN_RATE_LABEL, '-12d']) if WIN_RATE_LABEL in df.index and '-12d' in df.columns else 0
    afterwin = float(df.loc[WIN_RATE_LABEL, '12d']) if WIN_RATE_LABEL in df.index and '12d' in df.columns else 0

    times = len(df) - 2
    
    effectivepart1 = (
        '為有效訊號'
        if ((pre > 0 and after > 0) or (pre < 0 and after < 0)) and (times > 10) and ((prewin + afterwin > 140) or (prewin + afterwin < 60))
        else '不是有效訊號'
    )
    st.subheader(effectivepart1)

    # 並排：左表右圖
    col1, col2 = st.columns([1, 1])

    with col1:
        st.dataframe(resulttable1_list[0])

    with col2:
        if results_flat and results_flat[0].get('finalb1') is not None:
            finalb1_df = results_flat[0]['finalb1']
            if 'mean' in finalb1_df.columns:
                x = np.linspace(-31, 31, 31 + 31)
                y = finalb1_df['mean']
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(x, y, label='Final b1', color='darkblue')
                ax.axvline(0, color='grey', linestyle='--')
                xlim = (-15, 15)
                ax.set_xlim(xlim)
                ax.set_ylim(
                    bottom=y[(x >= xlim[0]) & (x <= xlim[1])].min() * 0.99,
                    top=y[(x >= xlim[0]) & (x <= xlim[1])].max() * 1.01)
                ax.set_xlabel('Months')
                ax.set_ylabel('Index')
                st.pyplot(fig, use_container_width=True)
else:
    st.info("第一段分析沒有可顯示的結果。")
    st.divider()


# ===== 第二段分析：breath / breath.shift(12) =====
st.subheader("年增率版本")
resulttable2_list = [r['resulttable2'] for r in results_flat if r.get('resulttable2') is not None]

if resulttable2_list:
    df = resulttable2_list[0]
    WIN_RATE_LABEL = "勝率"

    pre = float(df.loc['mean', '-12d']) - 100 if 'mean' in df.index and '-12d' in df.columns else 0
    after = float(df.loc['mean', '12d']) - 100 if 'mean' in df.index and '12d' in df.columns else 0
    prewin = float(df.loc[WIN_RATE_LABEL, '-12d']) if WIN_RATE_LABEL in df.index and '-12d' in df.columns else 0
    afterwin = float(df.loc[WIN_RATE_LABEL, '12d']) if WIN_RATE_LABEL in df.index and '12d' in df.columns else 0

    times = len(df) - 2

    effectivepart2 = (
        '為有效訊號'
        if ((pre > 0 and after > 0) or (pre < 0 and after < 0)) and (times > 10) and ((prewin + afterwin > 140) or (prewin + afterwin < 60))
        else '不是有效訊號'
    )
    st.subheader(effectivepart2)

    # 並排：左表右圖
    col1, col2 = st.columns([1, 1])

    with col1:
        st.dataframe(resulttable2_list[0])

    with col2:
        if results_flat and results_flat[0].get('finalb2') is not None:
            finalb2_df = results_flat[0]['finalb2']
            if 'mean' in finalb2_df.columns:
                x = np.linspace(-31, 31, 31 + 31)
                y = finalb2_df['mean']
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(x, y, label='Final b2', color='darkblue')
                ax.axvline(0, color='grey', linestyle='--')
                xlim = (-15, 15)
                ax.set_xlim(xlim)
                ax.set_ylim(
                    bottom=y[(x >= xlim[0]) & (x <= xlim[1])].min() * 0.99,
                    top=y[(x >= xlim[0]) & (x <= xlim[1])].max() * 1.01)
                ax.set_xlabel('Months')
                ax.set_ylabel('Index')
                st.pyplot(fig, use_container_width=True)
else:
    st.info("第二段分析沒有可顯示的結果。")


# ===== Plot by series_ids_text: Levels & YoY (brush to set x-range; y auto-rescales) =====
st.divider()
st.subheader("Each breath series: Levels (rolling mean ±σ) and YoY (brush to set time window)")

import altair as alt
alt.data_transformers.disable_max_rows()

sigma_levels = [0.5, 1.0, 1.5, 2.0]

def levels_chart_with_brush(s: pd.Series, sid: int, name: str):
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

    # melt to long format
    long_levels = df_levels.melt("Date", var_name="Series", value_name="Value").dropna()

    # brush selection on x (time)
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
        .properties(title=f"{name} ({sid}) | {winrolling_value}-period rolling mean ±σ", height=320)
    )

    lower = (
        alt.Chart(df_levels)
        .mark_area(opacity=0.4)
        .encode(x=alt.X("Date:T", title=""), y=alt.Y("Level:Q", title=""))
        .properties(height=60)
        .add_selection(brush)
    )

    return alt.vconcat(upper, lower).resolve_scale(y="independent")

def yoy_chart_with_brush(s: pd.Series, sid: int, name: str):
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
        .properties(title=f"{name} ({sid}) | YoY (%) with {winrolling_value}-period rolling mean ±σ", height=320)
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

# 獲取選定的系列名稱
if 'selected_name' not in st.session_state:
    st.session_state.selected_name = series_names[0]

selected_name = st.session_state.selected_name
# 根據名稱找到 ID
sid = id_name_map[id_name_map['名稱'] == selected_name]['ID'].iloc[0]

df_target = mm(int(sid), "MS", f"series_{sid}", k)
if df_target is None or df_target.empty:
    st.info(f"No data for series {sid}, skipping.")
else:
    s = df_target.iloc[:, 0].astype(float)
    with st.expander(f"Series: {selected_name} ({sid})", expanded=True):
        colA, colB = st.columns(2)
        with colA:
            st.altair_chart(levels_chart_with_brush(s, sid, selected_name), use_container_width=True)
        with colB:
            st.altair_chart(yoy_chart_with_brush(s, sid, selected_name), use_container_width=True)
