from flask import Flask, render_template, request, session, redirect, url_for, flash, send_from_directory
import pandas as pd
from kmodes.kprototypes import KPrototypes
import joblib
import os

app = Flask(__name__)


app.config['SQLALCHEMY_DATABASE_URI'] = 'postgres://u8lhv7k55jpqkn:p63e09a9ae211b1e1050e982500120a7665d41d5cd5363bc0a1314fb092a1c3aa@c5flugvup2318r.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/d7812gfl00q44o'



# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'



# Load the initial patient data
patients = pd.read_csv('oncosupport_dataset.csv')

# Simplified user storage
users = {}
user_storage_file = 'users.csv'
if not os.path.exists(user_storage_file):
    with open(user_storage_file, 'w') as f:
        f.write('username,password\n')

def load_users():
    global users
    user_data = pd.read_csv(user_storage_file)
    for index, row in user_data.iterrows():
        users[row['username']] = row['password']

def save_user(username, password):
    with open(user_storage_file, 'a') as f:
        f.write(f'{username},{password}\n')

load_users()

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
        if username in users:
            flash('Username already exists')
        else:
            users[username] = password
            save_user(username, password)
            session['user'] = username
            flash('Account created successfully')
            return redirect(url_for('login'))
    return render_template('signup.html')



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['user'] = username
            # Check if user already filled out the form and has a cluster
            user_data_file = f"user_data_{username}.pkl"
            if os.path.exists(user_data_file):
                with open(user_data_file, 'rb') as f:
                    user_data = joblib.load(f)
                    session.update(user_data)
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
                'Full Name': request.form['name'],
                'Gender': request.form['gender'],
                'Age': int(request.form['age']),
                'Ethnicity': request.form['ethnicity'],
                'Diagnosis by Location': request.form['diagnosis_by_location'],
                'Cancer Diagnosis': request.form['cancer_diagnosis'],
                'Cancer Stage': int(request.form['cancer_stage']),
                'Time with Cancer (months)': int(request.form['time_with_cancer']),
                'Primary Speaking Language': request.form['primary_language'],
                'Parenting Situation': request.form['parenting_situation'],
                'Personal Concern': request.form['personal_concern'],
                'Hospital': request.form['hospital'],
                'Phone Number': request.form['phone_number'],
                'Email': request.form['email'],
                'Username': session['user']
            }

            global patients
            patients = pd.concat([patients, pd.DataFrame([form_data])], ignore_index=True)
            
            # Perform clustering
            hospital_patients = patients[patients['Hospital'] == form_data['Hospital']]
            categorical_cols = [
                'Gender', 
                'Ethnicity', 
                'Diagnosis by Location', 
                'Cancer Diagnosis', 
                'Primary Speaking Language', 
                'Parenting Situation', 
                'Personal Concern'
            ]
            numerical_cols = ['Age', 
                              'Cancer Stage', 
                              'Time with Cancer (months)']

            hospital_patients_combined = hospital_patients[numerical_cols + categorical_cols].values
            n_clusters = max(1, len(hospital_patients) // 15)  # Ensure at least one cluster
            kproto = KPrototypes(n_clusters=n_clusters, init='Cao', n_init=10, random_state=123)
            
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
            session['hospital'] = form_data['Hospital']
            
            # Save user data to a file
            user_data = {
                'cluster_data': cluster_data,
                'cluster': new_patient_cluster,
                'hospital': form_data['Hospital']
            }
            user_data_file = f"user_data_{session['user']}.pkl"
            with open(user_data_file, 'wb') as f:
                joblib.dump(user_data, f)
            
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
