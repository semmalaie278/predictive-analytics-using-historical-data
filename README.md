# Predictive Analytics Using Historical Data

A full-stack **Predictive Analytics Dashboard** built with Python (Flask) and a modern dark-mode frontend. Upload any CSV dataset, auto-clean it, explore patterns with EDA, train ML or ARIMA models, and forecast future trends — all in one guided 6-step wizard.

---

## 🚀 Features

| Feature | Details |
|---|---|
| **CSV Upload** | Drag-and-drop or browse — up to 50 MB |
| **Auto Preprocessing** | Missing values (median/mode/ffill), duplicate removal, date parsing, type conversion |
| **Exploratory Data Analysis** | Summary statistics, correlation heatmap, distribution histograms |
| **Smart Model Selection** | Auto-detects time-series (ARIMA) vs regression datasets |
| **Linear Regression** | Scikit-learn, 80/20 split, MAE / MSE / RMSE / R² metrics |
| **ARIMA Forecasting** | Statsmodels ARIMA(1,1,1), 95% confidence intervals |
| **Interactive Charts** | Plotly.js — historical overview, actual vs predicted, forecast with bands |
| **Future Forecast** | Slider to select 1–100 future periods |
| **CSV Download** | Download all prediction results as a CSV file |
| **Sample Datasets** | Bundled housing (regression) and stock prices (time-series) |

---

## 🗂️ Project Structure

```
predictive-analytics/
├── app.py                  # Flask app & REST API routes
├── config.py               # Configuration (paths, limits, model defaults)
├── requirements.txt        # Python dependencies
│
├── modules/
│   ├── preprocessor.py     # Data cleaning & type conversion
│   ├── eda.py              # EDA: stats, heatmap, distributions
│   ├── model_selector.py   # Auto-detect regression vs time-series
│   ├── regression.py       # Linear Regression pipeline
│   ├── timeseries.py       # ARIMA pipeline
│   └── visualizer.py       # Historical overview charts
│
├── templates/
│   └── index.html          # 6-step wizard SPA (Jinja2)
│
├── static/
│   ├── css/styles.css      # Dark-mode glassmorphism design system
│   └── js/app.js           # Frontend wizard logic & Plotly rendering
│
└── sample_data/
    ├── housing.csv          # Regression sample (35 rows)
    └── stock_prices.csv     # Time-series ARIMA sample (40 rows)
```

---

## ⚙️ Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/semmalaie278/predictive-analytics-using-historical-data.git
cd predictive-analytics-using-historical-data
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

### 4. Open your browser
```
http://localhost:5000
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve the SPA |
| `POST` | `/api/upload` | Upload a CSV file |
| `GET` | `/api/samples` | List sample datasets |
| `POST` | `/api/load_sample` | Load a bundled sample |
| `POST` | `/api/preprocess` | Clean & preprocess data |
| `POST` | `/api/eda` | Run EDA, return charts |
| `POST` | `/api/train` | Train model, return metrics |
| `POST` | `/api/forecast` | Forecast N future periods |
| `GET` | `/api/download/<file>` | Download prediction CSV |
| `GET` | `/api/columns` | List numeric columns |

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, Flask
- **ML/Stats**: Scikit-learn (Linear Regression), Statsmodels (ARIMA)
- **Data**: Pandas, NumPy
- **Visualization**: Plotly (server JSON → Plotly.js frontend render)
- **Frontend**: HTML5, Vanilla CSS, Vanilla JavaScript
- **Design**: Dark-mode glassmorphism, Inter font, animated step progress

---

## 📋 Requirements

```
flask
pandas
numpy
scikit-learn
statsmodels
plotly
werkzeug
```

---

## 📄 License

MIT License — free to use, modify, and distribute.
