from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.secret_key = 'student-dropout-risk-management-secret-key-2024'

# Global variable to store prediction history
prediction_history = []

def load_student_data():
    try:
        df = pd.read_csv('data_fixed.csv')
        print("✅ Loaded existing data_fixed.csv")
    except:
        print("📊 Creating sample data_fixed.csv...")
        # Create comprehensive sample data
        np.random.seed(42)
        n_students = 500
        
        data = {
            'Marital status': np.random.choice([1, 2], n_students, p=[0.8, 0.2]),
            'Application mode': np.random.choice([1, 15, 16, 17, 18, 39, 42, 53], n_students),
            'Application order': np.random.randint(1, 10, n_students),
            'Course': np.random.choice([171, 8014, 9070, 9085, 9130, 9238, 9254, 9500, 9556, 9670, 9773, 9853, 9991], n_students),
            'Daytime/evening attendance': np.random.choice([0, 1], n_students, p=[0.3, 0.7]),
            'Previous qualification': np.random.choice([1, 2, 3, 19, 42], n_students),
            'Previous qualification (grade)': np.random.uniform(80, 180, n_students),
            'Nationality': np.random.choice([1, 62, 100], n_students, p=[0.8, 0.15, 0.05]),
            'Mother\'s qualification': np.random.choice([1, 3, 19, 37, 38], n_students),
            'Father\'s qualification': np.random.choice([1, 3, 12, 19, 37, 38], n_students),
            'Mother\'s occupation': np.random.choice([3, 4, 5, 7, 9], n_students),
            'Father\'s occupation': np.random.choice([3, 4, 5, 7, 8, 9, 10], n_students),
            'Admission grade': np.random.uniform(100, 150, n_students),
            'Displaced': np.random.choice([0, 1], n_students, p=[0.7, 0.3]),
            'Educational special needs': np.random.choice([0, 1], n_students, p=[0.95, 0.05]),
            'Debtor': np.random.choice([0, 1], n_students, p=[0.8, 0.2]),
            'Tuition fees up to date': np.random.choice([0, 1], n_students, p=[0.3, 0.7]),
            'Gender': np.random.choice([0, 1], n_students, p=[0.45, 0.55]),
            'Scholarship holder': np.random.choice([0, 1], n_students, p=[0.7, 0.3]),
            'Age at enrollment': np.random.randint(17, 40, n_students),
            'International': np.random.choice([0, 1], n_students, p=[0.9, 0.1]),
            'Curricular units 1st sem (credited)': np.random.randint(0, 3, n_students),
            'Curricular units 1st sem (enrolled)': np.random.randint(5, 10, n_students),
            'Curricular units 1st sem (evaluations)': np.random.randint(0, 10, n_students),
            'Curricular units 1st sem (approved)': np.random.randint(0, 8, n_students),
            'Curricular units 1st sem (grade)': np.random.uniform(0, 20, n_students),
            'Curricular units 1st sem (without evaluations)': np.random.randint(0, 3, n_students),
            'Curricular units 2nd sem (credited)': np.random.randint(0, 3, n_students),
            'Curricular units 2nd sem (enrolled)': np.random.randint(5, 10, n_students),
            'Curricular units 2nd sem (evaluations)': np.random.randint(0, 10, n_students),
            'Curricular units 2nd sem (approved)': np.random.randint(0, 8, n_students),
            'Curricular units 2nd sem (grade)': np.random.uniform(0, 20, n_students),
            'Curricular units 2nd sem (without evaluations)': np.random.randint(0, 3, n_students),
            'Unemployment rate': np.random.uniform(8, 17, n_students),
            'Inflation rate': np.random.uniform(-1, 4, n_students),
            'GDP': np.random.uniform(-5, 3, n_students),
            'Target': np.random.choice(['Dropout', 'Graduate', 'Enrolled'], n_students, p=[0.2, 0.6, 0.2])
        }
        
        df = pd.DataFrame(data)
        df.to_csv('data_fixed.csv', index=False)
        print(f"✅ Created sample data with {n_students} students")
    
    # Calculate academic metrics
    df['success_rate'] = (df['Curricular units 1st sem (approved)'] + df['Curricular units 2nd sem (approved)']) / \
                        (df['Curricular units 1st sem (enrolled)'] + df['Curricular units 2nd sem (enrolled)'] + 0.001)
    df['success_rate'] = df['success_rate'].fillna(0)
    
    df['average_grade'] = (df['Curricular units 1st sem (grade)'] + df['Curricular units 2nd sem (grade)']) / 2
    df['average_grade'] = df['average_grade'].fillna(0)
    
    # Create faculty mapping
    course_faculty_map = {
        171: 'Technology', 8014: 'Nursing', 9070: 'Social Work', 9085: 'Tourism',
        9130: 'Journalism', 9238: 'Management', 9254: 'Economics', 9500: 'Computer Science',
        9556: 'Design', 9670: 'Marketing', 9773: 'Education', 9853: 'Law', 9991: 'Medicine'
    }
    df['faculty'] = df['Course'].map(course_faculty_map).fillna('General')
    
    # Create risk levels
    risk_mapping = {'Dropout': 'High', 'Enrolled': 'Medium', 'Graduate': 'Low'}
    df['dropout_risk'] = df['Target'].map(risk_mapping)
    
    # Generate student names and IDs
    df['name'] = 'Student ' + (df.index + 1).astype(str)
    df['student_id'] = 'S' + (df.index + 1).astype(str).str.zfill(6)
    
    return df

def train_model():
    print("🤖 Training AI model...")
    df = load_student_data()
    
    # Select features for model
    feature_columns = [
        'Age at enrollment', 'Admission grade', 'Gender', 'Debtor', 
        'Scholarship holder', 'Tuition fees up to date', 'Previous qualification (grade)',
        'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)',
        'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (approved)',
        'Unemployment rate'
    ]
    
    X = df[feature_columns].fillna(0)
    y = (df['Target'] == 'Dropout').astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'feature_columns': feature_columns}, f)
    
    print("✅ Model trained and saved successfully")
    return model, scaler, feature_columns

def load_model():
    if os.path.exists('model.pkl'):
        print("📁 Loading existing model...")
        with open('model.pkl', 'rb') as f:
            data = pickle.load(f)
            return data['model'], data['scaler'], data['feature_columns']
    else:
        print("🆕 No existing model found, training new model...")
        return train_model()

# User database
users = {
    'admin': {'password': 'admin123', 'role': 'admin'},
    'adviser': {'password': 'adviser123', 'role': 'adviser'},
    'test': {'password': 'test123', 'role': 'adviser'}
}

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in users and users[username]['password'] == password:
            session['username'] = username
            session['role'] = users[username]['role']
            flash('✅ Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('❌ Invalid credentials!', 'danger')
    
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Login - Student Dropout System</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); height: 100vh; }
            .login-card { background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); }
            .btn-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border: none; }
        </style>
    </head>
    <body class="d-flex align-items-center">
        <div class="container">
            <div class="row justify-content-center">
                <div class="col-md-5">
                    <div class="login-card p-4">
                        <div class="text-center mb-4">
                            <h2 class="text-primary">🎓</h2>
                            <h3 class="mb-0">Student Dropout System</h3>
                            <p class="text-muted">Risk Management Portal</p>
                        </div>
                        <form method="POST">
                            <div class="mb-3">
                                <label class="form-label">Username</label>
                                <input type="text" class="form-control" name="username" required>
                            </div>
                            <div class="mb-3">
                                <label class="form-label">Password</label>
                                <input type="password" class="form-control" name="password" required>
                            </div>
                            <button type="submit" class="btn btn-primary w-100 py-2">Login</button>
                        </form>
                        <div class="mt-3 text-center">
                            <small class="text-muted">Demo: admin/admin123 | adviser/adviser123 | test/test123</small>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    df = load_student_data()
    
    # Calculate statistics
    total_students = len(df)
    overall_avg_grade = df['average_grade'].mean()
    total_at_risk = len(df[df['dropout_risk'] == 'High'])
    risk_percentage = (total_at_risk / total_students) * 100
    
    # Faculty statistics
    faculty_stats = {}
    for faculty in df['faculty'].unique():
        faculty_data = df[df['faculty'] == faculty]
        faculty_stats[faculty] = {
            'total_students': len(faculty_data),
            'avg_grade': faculty_data['average_grade'].mean(),
            'at_risk_count': len(faculty_data[faculty_data['dropout_risk'] == 'High']),
            'risk_percentage': (len(faculty_data[faculty_data['dropout_risk'] == 'High']) / len(faculty_data)) * 100
        }
    
    # Recent high-risk students
    recent_high_risk = df[df['dropout_risk'] == 'High'][['student_id', 'name', 'faculty', 'average_grade', 'Age at enrollment']].head(5).to_dict('records')
    
    # Generate faculty cards HTML
    faculty_cards_html = ""
    for faculty, stats in faculty_stats.items():
        risk_class = "risk-high" if stats['risk_percentage'] > 20 else "risk-medium" if stats['risk_percentage'] > 10 else "risk-low"
        faculty_cards_html += f'''
        <div class="col-md-6 mb-3">
            <div class="card faculty-card {risk_class}">
                <div class="card-body">
                    <h6 class="card-title">{faculty}</h6>
                    <div class="row text-center">
                        <div class="col-4">
                            <div class="border-end">
                                <h4 class="text-primary mb-0">{stats['total_students']}</h4>
                                <small class="text-muted">Students</small>
                            </div>
                        </div>
                        <div class="col-4">
                            <div class="border-end">
                                <h4 class="text-success mb-0">{stats['avg_grade']:.1f}%</h4>
                                <small class="text-muted">Avg Grade</small>
                            </div>
                        </div>
                        <div class="col-4">
                            <h4 class="text-danger mb-0">{stats['at_risk_count']}</h4>
                            <small class="text-muted">At Risk</small>
                        </div>
                    </div>
                    <div class="mt-2">
                        <small class="text-muted">Risk Level: {stats['risk_percentage']:.1f}%</small>
                        <div class="progress mt-1">
                            <div class="progress-bar bg-danger" style="width: {stats['risk_percentage']}%"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        '''
    
    # Generate high-risk alerts HTML
    high_risk_alerts_html = ""
    if recent_high_risk:
        for student in recent_high_risk:
            high_risk_alerts_html += f'''
            <div class="list-group-item">
                <div class="d-flex w-100 justify-content-between">
                    <h6 class="mb-1">{student['name']}</h6>
                    <span class="badge bg-danger">High Risk</span>
                </div>
                <p class="mb-1 small">
                    {student['faculty']} • Age {student['Age at enrollment']}<br>
                    Grade: {student['average_grade']:.1f}%
                </p>
                <small class="text-muted">ID: {student['student_id']}</small>
                <div class="mt-2">
                    <a href="/student/{student['student_id']}" class="btn btn-sm btn-outline-primary">
                        <i class="fas fa-eye"></i> View Profile
                    </a>
                </div>
            </div>
            '''
    else:
        high_risk_alerts_html = '<div class="list-group-item text-center text-muted">No high-risk students</div>'
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dashboard - Student Dropout System</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            .stat-card {{ transition: transform 0.2s; }}
            .stat-card:hover {{ transform: translateY(-5px); }}
            .faculty-card {{ border-left: 4px solid #007bff; }}
            .risk-high {{ border-left-color: #dc3545; }}
            .risk-medium {{ border-left-color: #ffc107; }}
            .risk-low {{ border-left-color: #28a745; }}
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
            <div class="container">
                <a class="navbar-brand" href="#"><i class="fas fa-graduation-cap"></i> Dropout Risk Manager</a>
                <div class="navbar-nav ms-auto">
                    <span class="navbar-text me-3">
                        <i class="fas fa-user"></i> {session['username']} ({session['role']})
                    </span>
                    <a class="nav-link" href="/logout"><i class="fas fa-sign-out-alt"></i> Logout</a>
                </div>
            </div>
        </nav>
        
        <div class="container mt-4">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h2><i class="fas fa-tachometer-alt"></i> Dashboard Overview</h2>
                <div class="btn-group">
                    <a href="/students" class="btn btn-outline-primary"><i class="fas fa-users"></i> Students</a>
                    <a href="/predict" class="btn btn-outline-primary"><i class="fas fa-robot"></i> AI Prediction</a>
                    <a href="/batch-analysis" class="btn btn-outline-primary"><i class="fas fa-cogs"></i> Batch Analysis</a>
                    <a href="/analytics" class="btn btn-outline-primary"><i class="fas fa-chart-bar"></i> Analytics</a>
                </div>
            </div>

            <!-- Key Metrics -->
            <div class="row mb-4">
                <div class="col-md-3 mb-3">
                    <div class="card stat-card bg-primary text-white">
                        <div class="card-body">
                            <div class="d-flex justify-content-between">
                                <div>
                                    <h5 class="card-title">Total Students</h5>
                                    <h2 class="mb-0">{total_students}</h2>
                                </div>
                                <div class="align-self-center">
                                    <i class="fas fa-users fa-2x"></i>
                                </div>
                            </div>
                            <p class="card-text">Across all faculties</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 mb-3">
                    <div class="card stat-card bg-success text-white">
                        <div class="card-body">
                            <div class="d-flex justify-content-between">
                                <div>
                                    <h5 class="card-title">Average Grade</h5>
                                    <h2 class="mb-0">{overall_avg_grade:.1f}%</h2>
                                </div>
                                <div class="align-self-center">
                                    <i class="fas fa-chart-line fa-2x"></i>
                                </div>
                            </div>
                            <p class="card-text">Overall performance</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 mb-3">
                    <div class="card stat-card bg-warning text-white">
                        <div class="card-body">
                            <div class="d-flex justify-content-between">
                                <div>
                                    <h5 class="card-title">At Risk</h5>
                                    <h2 class="mb-0">{total_at_risk}</h2>
                                </div>
                                <div class="align-self-center">
                                    <i class="fas fa-exclamation-triangle fa-2x"></i>
                                </div>
                            </div>
                            <p class="card-text">Need intervention</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3 mb-3">
                    <div class="card stat-card bg-info text-white">
                        <div class="card-body">
                            <div class="d-flex justify-content-between">
                                <div>
                                    <h5 class="card-title">Risk %</h5>
                                    <h2 class="mb-0">{risk_percentage:.1f}%</h2>
                                </div>
                                <div class="align-self-center">
                                    <i class="fas fa-percentage fa-2x"></i>
                                </div>
                            </div>
                            <p class="card-text">Of total students</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <!-- Faculty Statistics -->
                <div class="col-md-8">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="mb-0"><i class="fas fa-university"></i> Faculty Performance</h5>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                {faculty_cards_html}
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Sidebar -->
                <div class="col-md-4">
                    <!-- High Risk Alerts -->
                    <div class="card mb-4">
                        <div class="card-header bg-warning text-white">
                            <h6 class="mb-0"><i class="fas fa-exclamation-triangle"></i> High Risk Students Alert</h6>
                        </div>
                        <div class="card-body p-0">
                            <div class="list-group list-group-flush">
                                {high_risk_alerts_html}
                            </div>
                        </div>
                    </div>

                    <!-- Quick Actions -->
                    <div class="card">
                        <div class="card-header">
                            <h6 class="mb-0"><i class="fas fa-bolt"></i> Quick Actions</h6>
                        </div>
                        <div class="card-body">
                            <div class="d-grid gap-2">
                                <a href="/predict" class="btn btn-primary btn-sm">
                                    <i class="fas fa-robot"></i> Single Prediction
                                </a>
                                <a href="/batch-analysis" class="btn btn-outline-primary btn-sm">
                                    <i class="fas fa-cogs"></i> Batch Analysis
                                </a>
                                <a href="/students" class="btn btn-outline-primary btn-sm">
                                    <i class="fas fa-search"></i> Search Students
                                </a>
                                <a href="/analytics" class="btn btn-outline-primary btn-sm">
                                    <i class="fas fa-chart-pie"></i> View Analytics
                                </a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    '''

@app.route('/students')
def students():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    df = load_student_data()
    
    # Handle search filters
    search_name = request.args.get('name', '')
    search_faculty = request.args.get('faculty', '')
    search_risk = request.args.get('risk', '')
    
    filtered_df = df.copy()
    
    if search_name:
        filtered_df = filtered_df[filtered_df['name'].str.contains(search_name, case=False)]
    if search_faculty:
        filtered_df = filtered_df[filtered_df['faculty'] == search_faculty]
    if search_risk:
        filtered_df = filtered_df[filtered_df['dropout_risk'] == search_risk]
    
    students_data = filtered_df.to_dict('records')
    
    # Generate faculty options for dropdown
    faculty_options = "".join([f'<option value="{faculty}" {"selected" if search_faculty == faculty else ""}>{faculty}</option>' for faculty in df['faculty'].unique()])
    
    # Generate students table rows - Show ALL students (no limit)
    students_rows = ""
    for s in students_data:  # Removed the limit - show all students
        grade_badge_class = "success" if s["average_grade"] >= 14 else "warning" if s["average_grade"] >= 10 else "danger"
        risk_badge_class = "danger" if s["dropout_risk"] == "High" else "warning" if s["dropout_risk"] == "Medium" else "success"
        risk_icon = "exclamation-triangle" if s["dropout_risk"] == "High" else "exclamation-circle" if s["dropout_risk"] == "Medium" else "check-circle"
        status_badge_class = "danger" if s["Target"] == "Dropout" else "success" if s["Target"] == "Graduate" else "info"
        
        students_rows += f'''
        <tr>
            <td><code>{s["student_id"]}</code></td>
            <td><strong>{s["name"]}</strong></td>
            <td>{s["faculty"]}</td>
            <td>{s["Age at enrollment"]}</td>
            <td>
                <span class="badge bg-{grade_badge_class}">
                    {s["average_grade"]:.1f}%
                </span>
            </td>
            <td>
                <span class="badge bg-{risk_badge_class}">
                    <i class="fas fa-{risk_icon}"></i>
                    {s["dropout_risk"]}
                </span>
            </td>
            <td>
                <span class="badge bg-{status_badge_class}">
                    {s["Target"]}
                </span>
            </td>
            <td>
                <a href="/student/{s["student_id"]}" class="btn btn-sm btn-info">
                    <i class="fas fa-eye"></i> View Profile
                </a>
            </td>
        </tr>
        '''
    
    if not students_rows:
        students_rows = "<tr><td colspan='8' class='text-center text-muted py-4'>No students found</td></tr>"
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Students - Student Dropout System</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            .table-responsive {{ max-height: 70vh; overflow-y: auto; }}
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
            <div class="container">
                <a class="navbar-brand" href="/dashboard"><i class="fas fa-graduation-cap"></i> Dropout Risk Manager</a>
                <div class="navbar-nav ms-auto">
                    <span class="navbar-text me-3">
                        <i class="fas fa-user"></i> {session['username']}
                    </span>
                    <a class="nav-link" href="/logout"><i class="fas fa-sign-out-alt"></i> Logout</a>
                </div>
            </div>
        </nav>
        
        <div class="container mt-4">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h2><i class="fas fa-users"></i> Student Management</h2>
                <a href="/predict" class="btn btn-primary"><i class="fas fa-robot"></i> AI Prediction</a>
            </div>

            <!-- Search Form -->
            <div class="card mb-4">
                <div class="card-body">
                    <form method="GET" class="row g-3">
                        <div class="col-md-4">
                            <input type="text" class="form-control" name="name" placeholder="Search by name" value="{search_name}">
                        </div>
                        <div class="col-md-3">
                            <select class="form-control" name="faculty">
                                <option value="">All Faculties</option>
                                {faculty_options}
                            </select>
                        </div>
                        <div class="col-md-3">
                            <select class="form-control" name="risk">
                                <option value="">All Risk Levels</option>
                                <option value="Low" {"selected" if search_risk == "Low" else ""}>Low Risk</option>
                                <option value="Medium" {"selected" if search_risk == "Medium" else ""}>Medium Risk</option>
                                <option value="High" {"selected" if search_risk == "High" else ""}>High Risk</option>
                            </select>
                        </div>
                        <div class="col-md-2">
                            <button type="submit" class="btn btn-primary w-100">Search</button>
                        </div>
                    </form>
                </div>
            </div>

            <!-- Students Table -->
            <div class="card">
                <div class="card-body">
                    <div class="table-responsive">
                        <table class="table table-striped table-hover">
                            <thead class="table-dark">
                                <tr>
                                    <th>Student ID</th>
                                    <th>Name</th>
                                    <th>Faculty</th>
                                    <th>Age</th>
                                    <th>Average Grade</th>
                                    <th>Risk Level</th>
                                    <th>Status</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {students_rows}
                            </tbody>
                        </table>
                    </div>
                    <div class="mt-3">
                        <small class="text-muted">Showing {len(students_data)} students</small>
                    </div>
                </div>
            </div>

            <div class="mt-3">
                <a href="/dashboard" class="btn btn-secondary"><i class="fas fa-arrow-left"></i> Back to Dashboard</a>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    '''

@app.route('/student/<student_id>')
def student_profile(student_id):
    if 'username' not in session:
        return redirect(url_for('login'))
    
    df = load_student_data()
    student_data = df[df['student_id'] == student_id]
    
    if student_data.empty:
        flash('❌ Student not found!', 'danger')
        return redirect(url_for('students'))
    
    student = student_data.iloc[0]
    
    # Academic performance data
    academic_data = {
        'Admission Grade': student['Admission grade'],
        '1st Semester Grade': student['Curricular units 1st sem (grade)'],
        '2nd Semester Grade': student['Curricular units 2nd sem (grade)'],
        '1st Semester Approved': student['Curricular units 1st sem (approved)'],
        '2nd Semester Approved': student['Curricular units 2nd sem (approved)'],
        'Success Rate': f"{student['success_rate']*100:.1f}%"
    }
    
    # Risk factors
    risk_factors = []
    if student['Debtor']:
        risk_factors.append('Has outstanding debt')
    if not student['Tuition fees up to date']:
        risk_factors.append('Tuition fees not up to date')
    if student['average_grade'] < 10:
        risk_factors.append('Low academic performance')
    if student['Curricular units 1st sem (approved)'] < 3:
        risk_factors.append('Low course completion rate')
    if not risk_factors:
        risk_factors.append('No major risk factors identified')
    
    # Recommended interventions
    interventions = []
    if student['dropout_risk'] == 'High':
        interventions.append('Immediate academic counseling session')
        interventions.append('Financial aid assessment')
        interventions.append('Weekly progress monitoring')
        interventions.append('Parent/guardian consultation')
    if student['average_grade'] < 10:
        interventions.append('Tutoring services referral')
        interventions.append('Study skills workshop')
        interventions.append('Academic probation review')
    if student['Debtor']:
        interventions.append('Financial counseling')
        interventions.append('Payment plan options')
    if not student['Tuition fees up to date']:
        interventions.append('Urgent tuition fee discussion')
        interventions.append('Payment deadline extension request')
    if student['Curricular units 1st sem (approved)'] < 3:
        interventions.append('Course load adjustment')
        interventions.append('Academic advisor meeting')
    if not interventions:
        interventions.append('Continue current academic support')
        interventions.append('Regular academic check-ins')
    
    # Generate risk factors HTML
    risk_factors_html = "".join([f'<li><i class="fas fa-exclamation-circle text-danger"></i> {factor}</li>' for factor in risk_factors])
    
    # Generate interventions HTML
    interventions_html = "".join([f'''
    <div class="list-group-item">
        <i class="fas fa-check-circle text-success me-2"></i>
        {intervention}
    </div>
    ''' for intervention in interventions])
    
    # Generate academic metrics HTML
    academic_metrics_html = ""
    for metric, value in academic_data.items():
        if "Grade" in metric:
            value_class = "success" if (isinstance(value, (int, float)) and value > 12) else "warning" if (isinstance(value, (int, float)) and value > 8) else "danger"
        else:
            value_class = "success"
        academic_metrics_html += f'''
        <div class="col-md-4 mb-3">
            <div class="card bg-light">
                <div class="card-body text-center">
                    <h6 class="card-title">{metric}</h6>
                    <h3 class="text-{value_class}">
                        {value}
                    </h3>
                </div>
            </div>
        </div>
        '''
    
    risk_header_class = "danger" if student["dropout_risk"] == "High" else "warning" if student["dropout_risk"] == "Medium" else "success"
    risk_text_class = "danger" if student["dropout_risk"] == "High" else "warning" if student["dropout_risk"] == "Medium" else "success"
    status_badge_class = "danger" if student["Target"] == "Dropout" else "success" if student["Target"] == "Graduate" else "info"
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>{student["name"]} - Student Profile</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
            <div class="container">
                <a class="navbar-brand" href="/dashboard"><i class="fas fa-graduation-cap"></i> Dropout Risk Manager</a>
                <div class="navbar-nav ms-auto">
                    <span class="navbar-text me-3">
                        <i class="fas fa-user"></i> {session['username']}
                    </span>
                    <a class="nav-link" href="/logout"><i class="fas fa-sign-out-alt"></i> Logout</a>
                </div>
            </div>
        </nav>
        
        <div class="container mt-4">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h2><i class="fas fa-user-graduate"></i> Student Profile</h2>
                <a href="/students" class="btn btn-secondary"><i class="fas fa-arrow-left"></i> Back to Students</a>
            </div>

            <div class="row">
                <!-- Student Information -->
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header bg-primary text-white">
                            <h5 class="mb-0"><i class="fas fa-info-circle"></i> Student Information</h5>
                        </div>
                        <div class="card-body">
                            <p><strong>ID:</strong> <code>{student["student_id"]}</code></p>
                            <p><strong>Name:</strong> {student["name"]}</p>
                            <p><strong>Faculty:</strong> {student["faculty"]}</p>
                            <p><strong>Course:</strong> {student["Course"]}</p>
                            <p><strong>Age:</strong> {student["Age at enrollment"]}</p>
                            <p><strong>Gender:</strong> {"Male" if student["Gender"] == 1 else "Female"}</p>
                            <p><strong>Status:</strong> 
                                <span class="badge bg-{status_badge_class}">
                                    {student["Target"]}
                                </span>
                            </p>
                        </div>
                    </div>

                    <!-- Risk Assessment -->
                    <div class="card mt-3">
                        <div class="card-header bg-{risk_header_class} text-white">
                            <h5 class="mb-0"><i class="fas fa-chart-line"></i> Risk Assessment</h5>
                        </div>
                        <div class="card-body">
                            <div class="text-center">
                                <h1 class="display-4 text-{risk_text_class}">
                                    {student["dropout_risk"]}
                                </h1>
                                <p class="text-muted">Dropout Risk Level</p>
                            </div>
                            <div class="mt-3">
                                <h6>Risk Factors:</h6>
                                <ul class="list-unstyled">
                                    {risk_factors_html}
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Academic Performance -->
                <div class="col-md-8">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="mb-0"><i class="fas fa-chart-bar"></i> Academic Performance</h5>
                        </div>
                        <div class="card-body">
                            <div class="row text-center">
                                {academic_metrics_html}
                            </div>
                        </div>
                    </div>

                    <!-- Recommended Interventions -->
                    <div class="card mt-3">
                        <div class="card-header bg-info text-white">
                            <h5 class="mb-0"><i class="fas fa-lightbulb"></i> Recommended Interventions</h5>
                        </div>
                        <div class="card-body">
                            <div class="list-group">
                                {interventions_html}
                            </div>
                        </div>
                    </div>

                    <!-- Additional Information -->
                    <div class="card mt-3">
                        <div class="card-header">
                            <h5 class="mb-0"><i class="fas fa-cogs"></i> Additional Information</h5>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <p><strong>Scholarship:</strong> {"Yes" if student["Scholarship holder"] else "No"}</p>
                                    <p><strong>Debtor:</strong> {"Yes" if student["Debtor"] else "No"}</p>
                                    <p><strong>Tuition Up-to-date:</strong> {"Yes" if student["Tuition fees up to date"] else "No"}</p>
                                </div>
                                <div class="col-md-6">
                                    <p><strong>International:</strong> {"Yes" if student["International"] else "No"}</p>
                                    <p><strong>Displaced:</strong> {"Yes" if student["Displaced"] else "No"}</p>
                                    <p><strong>Special Needs:</strong> {"Yes" if student["Educational special needs"] else "No"}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    '''

# ... [Keep the /predict route exactly as it was in the previous code] ...

@app.route('/batch-analysis', methods=['GET', 'POST'])
def batch_analysis():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            # Get uploaded file
            uploaded_file = request.files['file']
            if uploaded_file.filename != '':
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                
                # Load model and make predictions
                model, scaler, feature_columns = load_model()
                
                # Prepare features (handle missing columns)
                available_features = [col for col in feature_columns if col in df.columns]
                missing_features = [col for col in feature_columns if col not in df.columns]
                
                if len(available_features) < 8:  # Minimum required features
                    flash(f'❌ Insufficient features in CSV. Required: {feature_columns}', 'danger')
                    return redirect('/batch-analysis')
                
                # Fill missing features with default values
                for feature in missing_features:
                    if feature == 'Previous qualification (grade)':
                        df[feature] = df.get('Admission grade', 120)
                    elif feature in ['Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)']:
                        df[feature] = 12.0
                    elif feature in ['Curricular units 1st sem (approved)', 'Curricular units 2nd sem (approved)']:
                        df[feature] = 5
                    elif feature == 'Unemployment rate':
                        df[feature] = 10.0
                    else:
                        df[feature] = 0
                
                features = df[feature_columns].fillna(0)
                features_scaled = scaler.transform(features)
                
                # Make predictions
                predictions = model.predict(features_scaled)
                probabilities = model.predict_proba(features_scaled)[:, 1]
                
                # Add predictions to dataframe
                df['predicted_risk'] = ['High Risk' if p == 1 else 'Low Risk' for p in predictions]
                df['confidence'] = [prob if p == 1 else (1 - prob) for p, prob in zip(predictions, probabilities)]
                df['confidence'] = (df['confidence'] * 100).round(2)
                
                # Sort by risk level (High Risk first, then by confidence)
                df_sorted = df.sort_values(['predicted_risk', 'confidence'], ascending=[False, False])
                
                # Generate results table with student data - Show ALL students (no limit)
                results_html = ""
                for idx, row in df_sorted.iterrows():  # Removed the limit - show all students
                    risk_badge = "danger" if row['predicted_risk'] == "High Risk" else "success"
                    
                    # Get student information
                    student_id = row.get('student_id', f'STD{idx+1:04d}')
                    name = row.get('name', f'Student {idx+1}')
                    faculty = row.get('faculty', 'Unknown')
                    age = row.get('Age at enrollment', 'N/A')
                    admission_grade = row.get('Admission grade', 'N/A')
                    gender = "Male" if row.get('Gender', 0) == 1 else "Female"
                    
                    results_html += f'''
                    <tr>
                        <td><code>{student_id}</code></td>
                        <td><strong>{name}</strong></td>
                        <td>{faculty}</td>
                        <td>{age}</td>
                        <td>{admission_grade}</td>
                        <td>{gender}</td>
                        <td><span class="badge bg-{risk_badge}">{row['predicted_risk']}</span></td>
                        <td>{row['confidence']}%</td>
                    </tr>
                    '''
                
                # Calculate batch statistics
                total_students = len(df)
                high_risk_count = (df['predicted_risk'] == 'High Risk').sum()
                high_risk_percentage = (high_risk_count / total_students) * 100
                avg_confidence_high = df[df['predicted_risk'] == 'High Risk']['confidence'].mean()
                avg_confidence_low = df[df['predicted_risk'] == 'Low Risk']['confidence'].mean()
                
                return f'''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Batch Analysis Results - Student Dropout System</title>
                    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
                    <style>
                        .table-responsive {{ max-height: 70vh; overflow-y: auto; }}
                    </style>
                </head>
                <body>
                    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
                        <div class="container">
                            <a class="navbar-brand" href="/dashboard"><i class="fas fa-graduation-cap"></i> Dropout Risk Manager</a>
                            <div class="navbar-nav ms-auto">
                                <span class="navbar-text me-3">
                                    <i class="fas fa-user"></i> {session['username']}
                                </span>
                                <a class="nav-link" href="/logout"><i class="fas fa-sign-out-alt"></i> Logout</a>
                            </div>
                        </div>
                    </nav>
                    
                    <div class="container mt-4">
                        <div class="row">
                            <div class="col-12">
                                <div class="card">
                                    <div class="card-header bg-success text-white">
                                        <h4 class="mb-0"><i class="fas fa-cogs"></i> Batch Analysis Results</h4>
                                    </div>
                                    <div class="card-body">
                                        <div class="row mb-4">
                                            <div class="col-md-3">
                                                <div class="card bg-primary text-white">
                                                    <div class="card-body text-center">
                                                        <h3>{total_students}</h3>
                                                        <p>Total Students</p>
                                                    </div>
                                                </div>
                                            </div>
                                            <div class="col-md-3">
                                                <div class="card bg-danger text-white">
                                                    <div class="card-body text-center">
                                                        <h3>{high_risk_count}</h3>
                                                        <p>High Risk</p>
                                                    </div>
                                                </div>
                                            </div>
                                            <div class="col-md-3">
                                                <div class="card bg-warning text-white">
                                                    <div class="card-body text-center">
                                                        <h3>{high_risk_percentage:.1f}%</h3>
                                                        <p>Risk Percentage</p>
                                                    </div>
                                                </div>
                                            </div>
                                            <div class="col-md-3">
                                                <div class="card bg-info text-white">
                                                    <div class="card-body text-center">
                                                        <h3>{avg_confidence_high:.1f}%</h3>
                                                        <p>Avg High Risk Confidence</p>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                        
                                        <div class="alert alert-info">
                                            <i class="fas fa-info-circle"></i> 
                                            Results are sorted by risk level (High Risk first) and confidence score.
                                            Showing all {len(df_sorted)} students.
                                        </div>
                                        
                                        <h5>Student Predictions:</h5>
                                        <div class="table-responsive">
                                            <table class="table table-striped table-hover">
                                                <thead class="table-dark">
                                                    <tr>
                                                        <th>Student ID</th>
                                                        <th>Name</th>
                                                        <th>Faculty</th>
                                                        <th>Age</th>
                                                        <th>Admission Grade</th>
                                                        <th>Gender</th>
                                                        <th>Risk Level</th>
                                                        <th>Confidence</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {results_html}
                                                </tbody>
                                            </table>
                                        </div>
                                        
                                        <div class="mt-4">
                                            <a href="/batch-analysis" class="btn btn-primary">
                                                <i class="fas fa-redo"></i> New Batch Analysis
                                            </a>
                                            <a href="/dashboard" class="btn btn-secondary">
                                                <i class="fas fa-tachometer-alt"></i> Dashboard
                                            </a>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </body>
                </html>
                '''
            else:
                flash('❌ Please upload a CSV file', 'danger')
                return redirect('/batch-analysis')
                
        except Exception as e:
            flash(f'❌ Error processing file: {str(e)}', 'danger')
            return redirect('/batch-analysis')
    
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Batch Analysis - Student Dropout System</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
            <div class="container">
                <a class="navbar-brand" href="/dashboard"><i class="fas fa-graduation-cap"></i> Dropout Risk Manager</a>
                <div class="navbar-nav ms-auto">
                    <span class="navbar-text me-3">
                        <i class="fas fa-user"></i> ''' + session.get('username', '') + '''
                    </span>
                    <a class="nav-link" href="/logout"><i class="fas fa-sign-out-alt"></i> Logout</a>
                </div>
            </div>
        </nav>
        
        <div class="container mt-4">
            <div class="row justify-content-center">
                <div class="col-md-8">
                    <div class="card">
                        <div class="card-header bg-primary text-white">
                            <h4 class="mb-0"><i class="fas fa-cogs"></i> Batch Analysis</h4>
                        </div>
                        <div class="card-body">
                            <p class="text-muted">Upload a CSV file with student data to perform batch dropout risk analysis.</p>
                            
                            <div class="alert alert-info">
                                <h6><i class="fas fa-info-circle"></i> CSV Format Requirements:</h6>
                                <ul class="mb-0">
                                    <li>File must be in CSV format</li>
                                    <li><strong>Required columns:</strong> Age at enrollment, Admission grade, Gender, Debtor, Scholarship holder, Tuition fees up to date</li>
                                    <li><strong>Recommended columns:</strong> Previous qualification (grade), Curricular units 1st sem (grade), Curricular units 2nd sem (grade)</li>
                                    <li>Missing columns will be filled with default values</li>
                                </ul>
                            </div>
                            
                            <form method="POST" enctype="multipart/form-data">
                                <div class="mb-3">
                                    <label class="form-label">Upload CSV File</label>
                                    <input type="file" class="form-control" name="file" accept=".csv" required>
                                </div>
                                <button type="submit" class="btn btn-primary btn-lg w-100">
                                    <i class="fas fa-cogs"></i> Analyze Batch
                                </button>
                            </form>
                        </div>
                    </div>
                    
                    <div class="mt-3 text-center">
                        <a href="/dashboard" class="btn btn-secondary">
                            <i class="fas fa-arrow-left"></i> Back to Dashboard
                        </a>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''

# ... [Keep the /analytics and /logout routes exactly as they were] ...

if __name__ == '__main__':
    print("🚀 Starting Student Dropout Risk Management System...")
    print("📊 Loading data and training AI model...")
    
    # Initialize data and model
    load_student_data()
    load_model()
    
    print("✅ System initialized successfully!")
    print("🌐 Starting Flask server...")
    print("🔗 Access the application at: http://localhost:5000")
    print("👤 Demo accounts:")
    print("   - admin / admin123")
    print("   - adviser / adviser123") 
    print("   - test / test123")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
