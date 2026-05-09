# ⚙️ MechVision AI — Mechanical Diagram Intelligence System

An end-to-end system that uses **Claude AI** (vision) + **scikit-learn** ML
to extract structured data from engineering diagrams and deliver plain-language
business insights — all inside a Streamlit web app.

---

## 🗂 Project structure

```
mech_analyzer/
├── app.py                  ← Streamlit frontend (all UI)
├── utils/
│   ├── backend.py          ← Claude vision extraction (LangChain-style)
│   ├── ml_engine.py        ← Linear Regression / SVM / KNN
│   └── insights.py         ← Claude business insights generator
├── .streamlit/
│   └── secrets.toml        ← Put your API key here
└── requirements.txt
```

---

## ⚡ Quick start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your Anthropic API key
Edit `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "sk-ant-xxxxxxxxxxxxxxxx"
```

### 3. Run the app
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 🔄 How it works

| Step | What happens |
|------|-------------|
| **1. Upload** | User uploads a PNG/JPG engineering diagram |
| **2. Extract** | Claude claude-sonnet-4-20250514 vision reads every labeled component and returns structured JSON (ref number, name, subsystem, complexity score, connections, size, etc.) |
| **3. ML model** | scikit-learn trains Linear Regression / SVM / KNN on the extracted features. Returns R², RMSE, MAE, predictions vs actuals, and feature importance |
| **4. Insights** | Claude generates 4 business insights, an executive summary table, and 4 prioritised action items — all in plain English |

---

## 🤖 Models supported

| Model | Best for | Notes |
|-------|----------|-------|
| **Linear Regression** | Interpretable results, small datasets | Shows coefficients & direction |
| **SVM (RBF kernel)** | Non-linear relationships | Good generalisation |
| **KNN** | Pattern matching, no assumptions | Sensitive to scale (auto-scaled) |

---

## 📊 Output tables

- **Component table** — every labeled part with subsystem, complexity, connections, size
- **Subsystem summary** — grouped stats with risk flags (🔴 🟡 🟢)
- **Predictions vs actual** — with % error per component, colour-coded
- **Feature importance** — which attributes drive complexity most
- **Executive summary** — 5-row table for boardroom presentation
- **Action items** — prioritised, plain-English recommendations

---

## 🔑 Without an API key

The app works in **demo mode** using the Valve Patent Design (Fig. 2) dataset
with 18 pre-extracted components. All ML and insights features work fully.

---

## 📦 Dependencies

- `streamlit` — web UI
- `anthropic` — Claude API (vision + text)
- `scikit-learn` — ML models + metrics
- `pandas` / `numpy` — data manipulation
