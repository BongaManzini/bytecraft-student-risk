# ByteCraft - Student Dropout & Academic Risk Management System 🎓

ByteCraft is an intelligent web application and Machine Learning platform designed to predict and manage student academic risk (Low, Medium, High) using Random Forest classification based on demographic, academic, and socioeconomic indicators.

## 🚀 Features

- **AI Dropout Risk Prediction**: Real-time risk evaluation using pre-trained Random Forest model (`model.pkl`).
- **Batch CSV Upload**: Upload custom student CSV files for automated bulk risk assessments.
- **Analytics & Dashboards**: Visual charts for risk distribution, academic grade comparisons, and demographic breakdown.
- **Student Profile Management**: Search, filter, and review individual student profiles and intervention recommendations.
- **Alerts & Reports**: Automated flagging for high-risk students needing academic interventions.
- **Model Transparency**: Dedicated page detailing ML model features, accuracy, and dataset structure.

---

## 🛠️ Project Structure

```
ByteCraft/
├── app.py              # Main Flask application
├── model.pkl           # Trained Random Forest ML model
├── data_fixed.csv      # Student dataset
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules (.venv, __pycache__, etc.)
├── static/             # CSS, JS, and image assets
│   ├── css/
│   ├── js/
│   └── images/
└── templates/          # HTML Jinja templates
    ├── home.html
    ├── dashboard.html
    ├── student_profile.html
    ├── upload_csv.html
    ├── upload.result.html
    ├── analytics.html
    ├── alerts.html
    ├── reports.html
    ├── students.html
    └── model_info.html
```

---

## 💻 Quick Start (Virtual Environment Setup)

### 1. Prerequisites
Ensure Python 3.8+ is installed on your system.

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
```

### 3. Activate Virtual Environment

- **Linux / macOS**:
  ```bash
  source .venv/bin/activate
  ```
- **Windows (PowerShell)**:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Application
```bash
python app.py
```

### 6. Access the Web Interface
Open your browser and navigate to:
```
http://localhost:5000
```
