# Rolling Mean Demo（Streamlit）

這個範例 App 可用隨機資料或 MacroMicro API 計算 rolling mean，並支援線性變換 `value = value * a + b` 與事件偵測。

## 快速開始（本機）
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Secrets（使用 MacroMicro API 才需要）
建立 `.streamlit/secrets.toml`，內容如下（**不要把真實金鑰提交到公開 repo**）：
```toml
MACROMICRO_API_KEY = "your_real_key_here"
```

## 部署到 Streamlit Community Cloud
1. 將此專案推到 GitHub。
2. 新增 App 並指向 `app.py`，Python 版本會根據 `runtime.txt` 使用 **3.11**。
3. 在「Secrets」設定 `MACROMICRO_API_KEY`。
4. 首次打開可先選 **Random demo** 測試。

## 相依套件與環境
- 套件見 `requirements.txt`
- Python 版本見 `runtime.txt`
