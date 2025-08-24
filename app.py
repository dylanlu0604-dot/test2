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

with st.sidebar:
    st.header("資料來源與參數設定")
    
    # 觸發邏輯選擇：Greater / Smaller
    trigger_mode = st.radio("觸發邏輯", ["Greater", "Smaller"], horizontal=True)

    series_ids_text = st.text_input("breath series IDs（逗號分隔）", "10000")
    assetid = st.number_input("index series ID (assetid)", min_value=0, value=0, step=1)
    api_key = st.text_input(
        "MacroMicro API Key（留空則使用 st.secrets 或環境變數）",
        value=st.secrets.get("MACROMICRO_API_KEY", os.environ.get("MACROMICRO_API_KEY", "")),
        type="password"
    )

    # 單選的 std 和 rolling 視窗（這些值在兩段分析中共用）
    std_choices = [0.5, 1.0, 1.5, 2.0]
    std_value = st.selectbox("標準差門檻", options=std_choices, index=1)

    roll_choices = [6, 12, 24, 36, 48]
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
                finalb1["median"] = finalb1.mean(axis=1)

                offsets = [-12, -6, 0, 6, 12]
                table1 = pd.concat([finalb1.iloc[timepast + off] for off in offsets], axis=1)
                table1.columns = [f"{off}d" for off in offsets]  # 仍沿用 d 命名
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
            "series_id": series_id, "std": std_value, "winrolling": winrolling_value,
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




df = resulttable1_list[0]  # 或換成 resulttable1_list[0] 做 part1

pre   = float(df.loc['median', '-12d']) - 100
after = float(df.loc['median', '12d'])  - 100
times = len(df) - 2  # 扣掉 勝率 + median

effectivepart1 = '為有效訊號' if (pre - 1) * (after - 1) > 0 and times > 10 else '不是有效訊號'



st.subheader(effectivepart1)


# 並排：左表右圖
col1, col2 = st.columns([1, 1])

with col1:
    if resulttable1_list:
        st.dataframe(resulttable1_list[0])  # 左邊顯示表格





with col2:
    if results_flat and results_flat[0].get('finalb1') is not None:
        # 右邊畫圖
        x = np.linspace(-31, 31, 31 + 31)
        y = results_flat[0]['finalb1']['median']
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(x, y, label='Final b1', color='darkgreen')
        ax.axvline(0, color='grey', linestyle='--')
        ax.set_xlim(-15, 15)
        ax.set_ylim(
            bottom=float(np.min(y) * 1),
            top=float(np.max(y) * 1)
        )
        ax.set_xlabel('Months')
        ax.set_ylabel('Index')
        st.pyplot(fig, use_container_width=True)

st.divider()


# ===== 第二段分析：breath / breath.shift(12) =====
st.subheader("年增率版本")
resulttable2_list = [r['resulttable2'] for r in results_flat if r.get('resulttable2') is not None]



df = resulttable2_list[0]  # 或換成 resulttable1_list[0] 做 part1

pre   = float(df.loc['median', '-12d']) - 100
after = float(df.loc['median', '12d'])  - 100
times = len(df) - 2  # 扣掉 勝率 + median

effectivepart2 = '為有效訊號' if (pre - 1) * (after - 1) > 0 and times > 10 else '不是有效訊號'


st.subheader(effectivepart2)


# 並排：左表右圖
col1, col2 = st.columns([1, 1])

with col1:
    if resulttable2_list:
        st.dataframe(resulttable2_list[0])  # 左邊顯示表格

with col2:
    if results_flat and results_flat[0].get('finalb2') is not None:
        # 右邊畫圖
        x = np.linspace(-31, 31, 31 + 31)
        y = results_flat[0]['finalb2']['median']
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(x, y, label='Final b2', color='darkblue')
        ax.axvline(0, color='grey', linestyle='--')
        ax.set_xlim(-15, 15)
        ax.set_ylim(
            bottom=float(np.min(y) * 1),
            top=float(np.max(y) * 1)
        )
        ax.set_xlabel('Months')
        ax.set_ylabel('Index')
        st.pyplot(fig, use_container_width=True)
