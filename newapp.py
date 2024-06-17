from flask import Flask, render_template, request, session, redirect, url_for, flash, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import pandas as pd
from kmodes.kprototypes import KPrototypes
import joblib
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a real secret key

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///oncosupport.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    form_submitted = db.Column(db.Boolean, default=False)
    cluster = db.Column(db.Integer)
    hospital = db.Column(db.String(120))

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    ethnicity = db.Column(db.String(50), nullable=False)
    diagnosis_by_location = db.Column(db.String(100), nullable=False)
    cancer_diagnosis = db.Column(db.String(100), nullable=False)
    cancer_stage = db.Column(db.Integer, nullable=False)
    time_with_cancer = db.Column(db.Integer, nullable=False)
    primary_language = db.Column(db.String(50), nullable=False)
    parenting_situation = db.Column(db.String(50), nullable=False)
    personal_concern = db.Column(db.String(100), nullable=False)
    hospital = db.Column(db.String(120), nullable=False)
    phone_number = db.Column(db.String(20))
    email = db.Column(db.String(120))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    cluster = db.Column(db.Integer)

# Load the initial patient data
patients = pd.read_csv('oncosupport_dataset.csv')

# Support Group Supervisors
supervisors = {
    'UC Health': {
        1: 'Andrew Smith',
        2: 'David Garcia',
        3: 'Alex Chen',
        4: 'Elizabeth Lin',
        5: 'Katie Wilson',
        6: 'Ali Stewart',
        7: 'Michael Thompson',
        8: 'Ryan Millay'
    },
    'Cleveland Clinic': {
        1: 'Andrew Smith',
        2: 'David Garcia',
        3: 'Alex Chen',
        4: 'Michael Thompson',
        5: 'Elizabeth Lin',
        6: 'Katie Wilson',
        7: 'Ali Stewart',
        8: 'Ryan Millay'
    },
    'Massachusetts General Hospital': {
        1: 'Andrew Smith',
        2: 'David Garcia',
        3: 'Alex Chen',
        4: 'Michael Thompson',
        5: 'Elizabeth Lin',
        6: 'Katie Wilson',
        7: 'Ali Stewart',
        8: 'Ryan Millay'
    }
}

@app.route('/')
def home():
    return redirect(url_for('signup'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password != confirm_password:
            flash('Passwords do not match')
            return render_template('signup.html')
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
        else:
            new_user = User(username=username, password=password)
            db.session.add(new_user)
            db.session.commit()
            session['user'] = username
            flash('Account created successfully')
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['user'] = username
            # Check if user already filled out the form and has a cluster
            if user.form_submitted:
                session['cluster_data'] = Patient.query.filter_by(user_id=user.id, cluster=user.cluster).all()
                session['cluster'] = user.cluster
                session['hospital'] = user.hospital
                return redirect(url_for('results'))
            return redirect(url_for('index'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('cluster', None)
    session.pop('cluster_data', None)
    session.pop('hospital', None)
    return redirect(url_for('signup'))

@app.route('/index', methods=['GET', 'POST'])
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            # Extract form data
            form_data = {
                'full_name': request.form['name'],
                'gender': request.form['gender'],
                'age': int(request.form['age']),
                'ethnicity': request.form['ethnicity'],
                'diagnosis_by_location': request.form['diagnosis_by_location'],
                'cancer_diagnosis': request.form['cancer_diagnosis'],
                'cancer_stage': int(request.form['cancer_stage']),
                'time_with_cancer': int(request.form['time_with_cancer']),
                'primary_language': request.form['primary_language'],
                'parenting_situation': request.form['parenting_situation'],
                'personal_concern': request.form['personal_concern'],
                'hospital': request.form['hospital'],
                'phone_number': request.form['phone_number'],
                'email': request.form['email'],
                'user_id': User.query.filter_by(username=session['user']).first().id
            }

            global patients
            patients = pd.concat([patients, pd.DataFrame([form_data])], ignore_index=True)
            
            # Perform clustering
            hospital_patients = patients[patients['hospital'] == form_data['hospital']]
            categorical_cols = [
                'gender', 
                'ethnicity', 
                'diagnosis_by_location', 
                'cancer_diagnosis', 
                'primary_language', 
                'parenting_situation', 
                'personal_concern'
            ]
            numerical_cols = ['age', 
                              'cancer_stage', 
                              'time_with_cancer']

            hospital_patients_combined = hospital_patients[numerical_cols + categorical_cols].values
            n_clusters = max(1, len(hospital_patients) // 15)  # Ensure at least one cluster
            kproto = KPrototypes(n_clusters=n_clusters, init='Cao', n_init=10, random_state=0)
            clusters = kproto.fit_predict(hospital_patients_combined, categorical=list(range(len(numerical_cols), len(numerical_cols) + len(categorical_cols))))
            hospital_patients['Cluster'] = clusters

            # Save the model (optional, if you need to reuse it)
            model_path = 'kproto_model.pkl'
            joblib.dump(kproto, model_path)

            # Determine the cluster of the new patient
            new_patient_cluster = int(hospital_patients.iloc[-1]['Cluster'])
            cluster_data = hospital_patients[hospital_patients['Cluster'] == new_patient_cluster].to_dict(orient='records')

            # Store results in session
            session['cluster_data'] = cluster_data
            session['cluster'] = new_patient_cluster
            session['hospital'] = form_data['hospital']
            
            # Save form submission data to the database
            new_patient = Patient(**form_data, cluster=new_patient_cluster)
            db.session.add(new_patient)
            user = User.query.filter_by(username=session['user']).first()
            user.form_submitted = True
            user.cluster = new_patient_cluster
            user.hospital = form_data['hospital']
            db.session.commit()
            
            return redirect(url_for('results'))
        except Exception as e:
            print(f"Error during form processing: {e}")
            flash(f"An error occurred: {e}")
            return redirect(url_for('index'))
    
    return render_template('index.html')

@app.route('/results')
def results():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Retrieve user's cluster results from session
    cluster_data = session.get('cluster_data', [])
    cluster = session.get('cluster', None)
    hospital = session.get('hospital', None)
    supervisor = supervisors.get(hospital, {}).get(cluster, 'No supervisor assigned')

    return render_template('results.html', cluster_data=cluster_data, cluster=cluster, hospital=hospital, supervisor=supervisor)

@app.route('/download')
def download():
    return send_from_directory(directory='/Users/siddhantnagar/Downloads/OncoSupport Phase I Materials', path='Support Group Rules and Guidelines.pdf', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
