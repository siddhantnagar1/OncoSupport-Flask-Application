import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from kmodes.kprototypes import KPrototypes
import joblib
app = Flask(__name__)
# Assuming the configuration for SQL and CSV path is correct
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgres://stbimxrfwilvqi:df442b27bc746243558eb6d7ca85a026e1d348db6e0abdc7aa14f663142344bb@ec2-34-236-103-63.compute-1.amazonaws.com:5432/d8vj75oq6sunk6'
# Load the initial patient data

patients = pd.read_csv('oncosupport_dataset.csv')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/cluster_links')
def cluster_links():
    return render_template('links.html')


def get_predicted_cluster(cluster_details):
    # Assuming cluster_details is a list of lists (each sublist is a cluster with patient records)
    # Determine the largest cluster
    if not cluster_details:
        return None
    largest_cluster = max(cluster_details, key=len)
    predicted_cluster_index = cluster_details.index(largest_cluster)  # get the index of the largest cluster
    return predicted_cluster_index

@app.route('/all-patients')
def all_patients():
    # Assuming 'patients' is a list of dictionaries, each containing patient data
    patients = pd.read_csv('oncosupport_dataset.csv')
    return render_template('results.html', patients=patients)
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
        print ("Initial State:")
        print(f"Number of patients: {len(patients)}")
        patients = pd.concat([patients, pd.DataFrame([new_entry])], ignore_index=True)
        print ("One")
        cluster_details = cluster_patients_by_hospital(hospital)
        print("Cluster Data to be sent:", cluster_details)
        
        # Perform clustering by hospital
        if hospital:
            cluster_details = cluster_patients_by_hospital(hospital)
            if cluster_details and len(cluster_details) > 0:
                predicted_cluster = get_predicted_cluster(cluster_details)
                return render_template('results.html', cluster_data=cluster_details, cluster=predicted_cluster)
            else:
                return render_template('results.html', message="Not enough data to form clusters.")
        else:
            return render_template('results.html', message="No hospital specified.")
    except Exception as e:
        return render_template('error.html', error=str(e))  # Consider creating an error.html template for better error handling

print ("Two")

def cluster_patients_by_hospital(hospital_name):
    # Filter patients for a specific hospital
    hospital_patients = patients[patients['Hospital'] == hospital_name]
    print("Data before clustering:", hospital_patients)
    print(f"Number of patients in {hospital_name}: {len(hospital_patients)}")
    print ("Three")

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
    
    print ("Four")

    # Preparing data for clustering
    hospital_patients_numerical = hospital_patients[numerical_cols].values
    hospital_patients_categorical = hospital_patients[categorical_cols].values
    hospital_patients_combined = np.hstack((hospital_patients_numerical, hospital_patients_categorical))
    print ("Five")

    # Perform clustering if enough data is available
    if len(hospital_patients) >= 50:  # Ensure there are enough patients
        print ("Six")
        kproto = KPrototypes(n_clusters=5, init='Cao', n_init=10, random_state=0)
        print ("Seven)")
        clusters = kproto.fit_predict(hospital_patients_combined, categorical=list(range(len(numerical_cols), len(numerical_cols) + len(categorical_cols))))
        print("After Clustering -- ")
        print(f"Clusters = {clusters}")
        print(f"Number of clusters = {len(set(clusters))}")
        
        hospital_patients['Cluster'] = clusters
        joblib.dump(kproto, 'kproto_model.pkl')
        print(f"Cluster assignment for {hospital_name}:")
        cluster_counts = hospital_patients['Cluster'].value_counts()
        print(cluster_counts)
        predicted_cluster = clusters[-1]  # Assuming you are interested in the cluster of the last added patient
       
        filtered_patients = hospital_patients[hospital_patients['Cluster'] == predicted_cluster]
        print ("Predicted Cluster:")
        print (predicted_cluster)
        # Return and render the template with the filtered patient data
        return render_template('results.html', cluster_data=filtered_patients.to_dict(orient='records'), cluster=predicted_cluster)                          
    else:
        return "Not enough data to form clusters."
    
if __name__ == '__main__':
    app.run(debug=True)
