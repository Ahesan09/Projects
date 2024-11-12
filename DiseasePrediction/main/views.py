from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm,UserCreationForm
from django.contrib.auth.models import User
import os
from django.shortcuts import render, HttpResponse
import pandas as pd
from DiseasePrediction import settings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
def home(request):
    return render(request, 'home.html')

@login_required
def health_prediction(request):
    return render(request,'health_prediction.html')

def loginaccount(request):
    if request.method == 'GET':
        return render(request, 'loginaccount.html',{'form':AuthenticationForm})
    else:
        user = authenticate(request,username=request.POST['username'],password=request.POST['password'])
        if user is None:
            return render(request,'loginaccount.html',{'form': AuthenticationForm(),'error': 'username and password do not match'})
        else:
            login(request,user)
            return render(request,'home.html')


def signupaccount(request):
    if request.method == "GET":
        return render(request,'signupaccount.html',{'form':UserCreationForm})
    else:
        if request.POST['password1'] == request.POST['password2']:
            user = User.objects.create_user(request.POST['username'],password=request.POST['password1'])
            user.save()
            login(request,user)
            return render(request,'home.html')
        else:
            return render(render,'signupaccount.html',{'form':UserCreationForm,'error':'Password do not match'})
        
def logoutaccount(request):
    logout(request)
    return render(request,'home.html')

def diabetesPage(request):
    return render(request,'diabetes.html')

def lungCancerPage(request):
    return render(request,'lungcancer.html')

def heartPage(request):
     return render(request,'heart.html')

def Diabetes_prediction(request):
    # Load the dataset
    file_path = os.path.join(settings.BASE_DIR, 'main/model/diabetes.csv')
    df = pd.read_csv(file_path)

    # Split data into input (x) and output (y)
    y = df[['Outcome']]
    x = df.iloc[:, :8]

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

    # Train the model
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    # Define required fields
    required_fields = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'Bmi', 'DiabetesPedigreeFunction', 'Age']

    # Check for missing or invalid input
    for field in required_fields:
        field_value = request.GET.get(field)
        if field_value is None or field_value.strip() == "" or float(field_value) < 0:
            outcome = 'Please enter valid data.'

    # Get input values from GET request
    PregnanciesInput = int(request.GET.get('Pregnancies', 0))  
    GlucoseInput = int(request.GET.get('Glucose', 0))
    BloodPressureInput = int(request.GET.get('BloodPressure', 0)) 
    SkinThicknessInput = int(request.GET.get('SkinThickness', 0))
    InsulinInput = int(request.GET.get('Insulin', 0))
    BmiInput = float(request.GET.get('Bmi', 0.0))
    DiabetesPedigreeFunctionInput = float(request.GET.get('DiabetesPedigreeFunction', 0.0))
    AgeInput = int(request.GET.get('Age', 0))

    # Reshape the input data to be 2D
    input_data = np.array([PregnanciesInput, GlucoseInput, BloodPressureInput, SkinThicknessInput, InsulinInput, BmiInput, DiabetesPedigreeFunctionInput, AgeInput]).reshape(1, -1)

    # Make prediction
    predict = lr.predict(input_data)
    predict = round(predict[0][0])  # Adjust the prediction output

    # Determine the 
    if predict >=0.5:
        outcome = "You are diabetic patient" 
    else:
        outcome = "You are not diabetic patient"

    # Render the result in the template
    return render(request, 'diabetes.html', {'outcome': outcome})


def lungCancerPrediction(request):
    # Load the dataset
    file_path = os.path.join(settings.BASE_DIR, 'main/model/survey lung cancer.csv')
    df = pd.read_csv(file_path)

    # Split data into input (x) and output (y)
    y = df[['LUNG_CANCER']]
    x = df.drop(columns=['LUNG_CANCER'])

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

    # Train the model
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    outcome = None

    if request.method == 'GET':
        # Retrieve input data from the request
        gender = request.GET.get('gender')
        age = request.GET.get('age')
        smoking = request.GET.get('smoking')
        yellow_fingers = request.GET.get('yellow_fingers')
        anxiety = request.GET.get('anxiety')
        peer_pressure = request.GET.get('peer_pressure')
        chronic_disease = request.GET.get('chronic_disease')
        fatigue = request.GET.get('fatigue')
        allergy = request.GET.get('allergy')
        wheezing = request.GET.get('wheezing')
        alcohol_consuming = request.GET.get('alcohol_consuming')
        coughing = request.GET.get('coughing')
        shortness_of_breath = request.GET.get('shortness_of_breath')
        swallowing_difficulty = request.GET.get('swallowing_difficulty')
        chest_pain = request.GET.get('chest_pain')

        # Check for None values and convert to appropriate types
        try:
                gender = 1 if gender == 'male' else 0
                age = int(age) if age else 0  # Use 0 or another default if age is None
                smoking = 1 if smoking == 'yes' else 0
                yellow_fingers = 1 if yellow_fingers == 'yes' else 0
                anxiety = 1 if anxiety == 'yes' else 0
                peer_pressure = 1 if peer_pressure == 'yes' else 0
                chronic_disease = 1 if chronic_disease == 'yes' else 0
                fatigue = 1 if fatigue == 'yes' else 0
                allergy = 1 if allergy == 'yes' else 0
                wheezing = 1 if wheezing == 'yes' else 0
                alcohol_consuming = 1 if alcohol_consuming == 'yes' else 0
                coughing = 1 if coughing == 'yes' else 0
                shortness_of_breath = 1 if shortness_of_breath == 'yes' else 0
                swallowing_difficulty = 1 if swallowing_difficulty == 'yes' else 0
                chest_pain = 1 if chest_pain == 'yes' else 0

                # Prepare input data for prediction
                input_data = np.array([gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
                                    chronic_disease, fatigue, allergy, wheezing, 
                                    alcohol_consuming, coughing, shortness_of_breath, 
                                    swallowing_difficulty, chest_pain]).reshape(1, -1)

                # Make prediction
                predict = lr.predict(input_data)
                predict = round(predict[0][0])  # Adjust the prediction output

                # Get the prediction result
                outcome = "Positive" if predict >= 0.5 else "Negative"

        except (ValueError, TypeError) as e:
                outcome = "Invalid input: " + str(e)

    return render(request, 'lungcancer.html', {'outcome': outcome})



def heartDiseasePrediction(request):
    # Load dataset
    file_path = os.path.join(settings.BASE_DIR, 'main/model/heart.csv')
    df = pd.read_csv(file_path)

    # Prepare the data for training
    y = df[['target']]
    x = df.drop(columns=['target'])

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

    # Train the Linear Regression model
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    # If a form is submitted, retrieve the input values
    if request.method == 'GET' and 'age' in request.GET:
        try:
            # Retrieve and explicitly convert inputs to appropriate types (float or int)
            age = float(request.GET.get('age'))
            sex = 1 if request.GET.get('sex') == 'Male' else 0  # Assuming Male = 1, Female = 0
            cp = float(request.GET.get('cp'))
            trestbps = float(request.GET.get('trestbps'))
            chol = float(request.GET.get('chol'))
            fbs = 1 if request.GET.get('fbs') == 'YES' else 0  # Assuming YES = 1, NO = 0
            restecg = float(request.GET.get('restecg'))
            thalach = float(request.GET.get('thalach'))
            exang = 1 if (request.GET.get('exang')) =='YES' else 0
            oldpeak = float(request.GET.get('oldpeak'))
            slope = float(request.GET.get('slope'))
            ca = float(request.GET.get('ca'))
            thal = float(request.GET.get('thal'))

            # Combine all input values into a NumPy array for compatibility with the model
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

            # Make prediction using the trained model
            prediction = lr.predict(input_data)

            # Convert prediction result to a readable form 
            outcome = 'Positive for Heart Disease' if prediction[0][0] >= 0.5 else 'Negative for Heart Disease'

        except ValueError as e:
            # Handle any value conversion issues
            outcome = "Invalid input: Please make sure all fields are filled with numeric values."

        # Render the template with the prediction outcome
        return render(request, 'heart.html', {'outcome': outcome})

    # Render the form if no prediction request
    return render(request, 'heart.html') 

