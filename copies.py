import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from kmodes.kprototypes import KPrototypes
import joblib

app = Flask(__name__)

# Load the initial patient data
patients = pd.read_csv('oncosupport_dataset.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/cluster_links')
def cluster_links():
    return render_template('links.html')

@app.route('/all-patients')
def all_patients():
    patients = pd.read_csv('oncosupport_dataset.csv')
    patients_list = patients.to_dict(orient='records')
    
    # Debug print statements
    print("Data read from CSV:")
    print(patients.head())  # Print the first few rows of the DataFrame to check the data
    print("Data being passed to the template:")
    print(patients_list[:5])  # Print the first few entries of the list of dictionaries
    
    return render_template('results.html', patients=patients_list)

@app.route('/cluster', methods=['POST'])
def cluster():
    try:
        # Extract and type convert form data safely
        name = request.form['name']
        gender = request.form['gender']
        age = int(request.form['age'])
        ethnicity = request.form['ethnicity']
        diagnosis_by_location = request.form['diagnosis_by_location']
        cancer_diagnosis = request.form['cancer_diagnosis']
        cancer_stage = int(request.form['cancer_stage'])
        time_with_cancer = int(request.form['time_with_cancer'])
        primary_language = request.form['primary_language']
        parenting_situation = request.form['parenting_situation']
        personal_concern = request.form['personal_concern']
        hospital = request.form['hospital']
        phone_number = request.form['phone_number']
        email = request.form['email']

        # Append the new entry to the DataFrame
        new_entry = {
            'Full Name': name, 
            'Gender': gender, 
            'Age': age, 
            'Ethnicity': ethnicity,
            'Diagnosis by Location': diagnosis_by_location, 
            'Cancer Diagnosis': cancer_diagnosis,
            'Cancer Stage': cancer_stage, 
            'Time with Cancer (months)': time_with_cancer,
            'Primary Speaking Language': primary_language, 
            'Parenting Situation': parenting_situation,
            'Personal Concern': personal_concern, 
            'Hospital': hospital,
            'Phone Number': phone_number, 
            'Email': email
        }

        global patients
        patients = pd.concat([patients, pd.DataFrame([new_entry])], ignore_index=True)

        # Perform clustering by hospital
        if hospital:
            cluster_data = cluster_patients_by_hospital(hospital)
            if not cluster_data.empty:
                predicted_cluster = cluster_data['Cluster'].iloc[-1]  # Get the cluster of the newly added patient
                cluster_members = cluster_data[cluster_data['Cluster'] == predicted_cluster]

                print(f"Predicted cluster: {predicted_cluster}")
                print(f"Cluster data (first 5): {cluster_members.head()}")

                return render_template('results.html', cluster_data=cluster_members.to_dict(orient='records'), cluster=predicted_cluster)
            else:
                return render_template('results.html', message="Not enough data to form clusters.")
        else:
            return render_template('results.html', message="No hospital specified.")
    except Exception as e:
        return render_template('error.html', error=str(e))

def cluster_patients_by_hospital(hospital_name):
    # Filter patients for a specific hospital
    hospital_patients = patients[patients['Hospital'] == hospital_name]
    print("Data before clustering:", hospital_patients)
    print(f"Number of patients in {hospital_name}: {len(hospital_patients)}")

    # Columns for clustering
    categorical_cols = ['Gender', 
                        'Ethnicity', 
                        'Diagnosis by Location', 
                        'Cancer Diagnosis', 
                        'Primary Speaking Language', 
                        'Parenting Situation', 
                        'Personal Concern']
    numerical_cols = ['Age', 
                      'Cancer Stage', 
                      'Time with Cancer (months)']
    
    # Preparing data for clustering
    hospital_patients_numerical = hospital_patients[numerical_cols].values
    hospital_patients_categorical = hospital_patients[categorical_cols].values
    hospital_patients_combined = np.hstack((hospital_patients_numerical, hospital_patients_categorical))

    # Debug statements to check data
    print("Hospital patients numerical data:")
    print(hospital_patients_numerical[:5])
    print("Hospital patients categorical data:")
    print(hospital_patients_categorical[:5])
    print("Combined data for clustering:")
    print(hospital_patients_combined[:5])

    # Perform clustering if enough data is available
    if len(hospital_patients) >= 50:  # Ensure there are enough patients
        kproto = KPrototypes(n_clusters=8, init='Cao', n_init=10, random_state=0)
        
        clusters = kproto.fit_predict(hospital_patients_combined, categorical=list(range(len(numerical_cols), len(numerical_cols) + len(categorical_cols))))
        print("After Clustering -- ")
        print(f"Clusters = {clusters}")
        print(f"Number of clusters = {len(set(clusters))}")
        
        hospital_patients['Cluster'] = clusters
        joblib.dump(kproto, 'kproto_model.pkl')
        print(f"Cluster assignment for {hospital_name}:")
        cluster_counts = hospital_patients['Cluster'].value_counts()
        print(cluster_counts)
        
        return hospital_patients
    else:
        return pd.DataFrame()  # Return an empty DataFrame if not enough data

if __name__ == '__main__':
    app.run(debug=True)
