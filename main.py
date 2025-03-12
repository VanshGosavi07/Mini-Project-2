import requests
import json
from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import ast
from flask import Flask, render_template, redirect, url_for, session, flash
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, SelectField
from wtforms.validators import DataRequired, Email, ValidationError
from wtforms import TextAreaField, FileField, SelectField
from wtforms.validators import DataRequired, Email, ValidationError, Length
from wtforms.fields import DateField
from flask_wtf.file import FileField, FileRequired, FileAllowed
import bcrypt
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
# SQLite Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'your_secret_key_here'

db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    contact_number = db.Column(db.String(15), nullable=False)
    date_of_birth = db.Column(db.String(10), nullable=False)
    city = db.Column(db.String(50), nullable=False)
    user_type = db.Column(db.String(10), nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    age = db.Column(db.Integer, nullable=False)


class RegisterForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired()])
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    contact_number = StringField("Contact Number", validators=[DataRequired()])
    date_of_birth = StringField("Date of Birth", validators=[DataRequired()])
    city = StringField("City", validators=[DataRequired()])
    user_type = SelectField("User Type", choices=[(
        'doctor', 'Doctor'), ('patient', 'Patient')], validators=[DataRequired()])
    gender = SelectField("Gender", choices=[('male', 'Male'), ('female', 'Female')], validators=[
                         DataRequired()])  # New field for gender
    submit = SubmitField("Register")

    def validate_email(self, field):
        if User.query.filter_by(email=field.data).first():
            raise ValidationError('Email Already Taken')


class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")

class DiagnosisForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    dob = DateField('Date of Birth', validators=[DataRequired()])
    disease_name = SelectField('Disease Name', 
                               choices=[('Breast Cancer', 'Breast Cancer')],
                               validators=[DataRequired()])
    ct_images = FileField('Upload CT Images', 
                          validators=[DataRequired(), FileAllowed(['png'], 'Images only!')])
    clinical_history = TextAreaField('Clinical History', validators=[DataRequired()])
    symptoms = StringField('Symptoms', validators=[DataRequired()])
    submit = SubmitField('Generate Report')

    def validate_name(self, field):
        if not field.data.isalpha():
            raise ValidationError('Name must contain only letters')


if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


# Function to Navigate home page
@app.route('/')
def home():
    return render_template('home.html')

# Function to Navigate to the register page
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        name = form.name.data
        email = form.email.data
        password = form.password.data
        contact_number = form.contact_number.data
        date_of_birth = form.date_of_birth.data
        city = form.city.data
        user_type = form.user_type.data
        gender = form.gender.data  # Get gender from form

        # Calculate age from date_of_birth
        dob_date = datetime.strptime(date_of_birth, '%Y-%m-%d')
        age = datetime.now().year - dob_date.year - ((datetime.now().month,
                           datetime.now().day) < (dob_date.month, dob_date.day))

        hashed_password = bcrypt.hashpw(
            password.encode('utf-8'), bcrypt.gensalt())

        new_user = User(name=name, email=email, password=hashed_password, contact_number=contact_number,
                        # Include gender and age
                        date_of_birth=date_of_birth, city=city, user_type=user_type, gender=gender, age=age)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))

    return render_template('register.html', form=form)

# Function to Navigate to the login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data

        user = User.query.filter_by(email=email).first()
        if user and bcrypt.checkpw(password.encode('utf-8'), user.password):
            session['user_id'] = user.id
            session['name']=user.name
            session['user_type']=user.user_type
            return redirect(url_for('form'))
        else:
            flash("Login failed. Please check your email and password")
            return redirect(url_for('login'))

    return render_template('login.html', form=form)

@app.route('/form', methods=['GET', 'POST'])
def form():
    form = DiagnosisForm()
    return render_template('form.html', form=form)

# Function to logout the user
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("You have been logged out successfully.")
    return redirect(url_for('login'))


# Function to Generate the Report from form Data
@app.route('/generate_report', methods=['POST'])
def generate_report():
    if request.method == 'POST':
        # Gather form data
        name = request.form['name']
        dob = request.form['dob']
        disease_name = request.form['disease_name']
        clinical_history = request.form['clinical_history']
        prepared_by = session.get('name')

        # Parse symptoms JSON
        symptoms_json = request.form.get('symptoms_json')
        if symptoms_json:
            symptoms = json.loads(symptoms_json)
        else:
            symptoms = []

        # Calculate age
        dob_date = datetime.strptime(dob, '%Y-%m-%d')
        age = datetime.now().year - dob_date.year - ((datetime.now().month,
                                                      datetime.now().day) < (dob_date.month, dob_date.day))

        # Handle file uploads
        ct_images = request.files.getlist('ct_images')
        image_paths = []
        for image in ct_images:
            if image:
                # Save to static/uploads and store relative path
                filename = image.filename
                image_path = os.path.join(
                    app.config['UPLOAD_FOLDER'], filename)
                image.save(image_path)
                # Store relative path for template
                relative_path = f'uploads/{filename}'
                image_paths.append(relative_path)

        # Call the model function
        results = model_predict(image_paths)
        diseases_level = [result[0] for result in results]
        new_image_paths = [result[1] for result in results]
        print(f'Diseases level: {diseases_level}')

        data = generate_data(name, age, disease_name, clinical_history, symptoms, new_image_paths, diseases_level)
        data = ast.literal_eval(data)  # Ensure data is a dictionary
        print(f"Generated data: {data}")

        # Pass data to the report template
        current_date = datetime.now().strftime('%Y-%m-%d')
        user_type = session.get('user_type')
        
        return render_template('report.html',user_type=user_type, name=name, dob=dob, age=age, disease_name=disease_name, clinical_history=clinical_history, symptoms=symptoms, prepared_by=prepared_by, image_paths=new_image_paths, diseases_level=diseases_level, data=data, current_date=current_date)

    return redirect(url_for('home'))

# Function to predict the cancer using the given image
def model_predict(image_paths):
    # Load your pre-trained model
    model = load_model('D:/Project/Mini Project 2/Modal/breast_cancer.keras')

    # Initialize an empty list to store predictions and new image paths
    results = []

    for img_path in image_paths:
        # Load the image using OpenCV
        img_full_path = os.path.join('static', img_path)
        if not os.path.exists(img_full_path):
            raise FileNotFoundError(f"File not found: {img_full_path}")
        img_cv = cv2.imread(img_full_path)

        # Make a copy for display
        img_copy = img_cv.copy()

        # Resize image to 224x224 as required by the CNN model
        img_resized = cv2.resize(img_cv, (224, 224))

        # Preprocess the image for the model
        img_array = image.img_to_array(img_resized) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make the prediction
        prediction = model.predict(img_array)
        if prediction[0][0] >= 0.5:  # Assuming binary classification with threshold 0.5
            label = "Cancer: Yes (Malignant)"
            color = (0, 0, 255)  # Red for cancer
        else:
            label = "Cancer: No (Non-Malignant)"
            color = (0, 255, 0)  # Green for no cancer

        # Draw a rectangle around the original image
        height, width, _ = img_copy.shape
        padding = 100  # Adjust for square size
        start_point = (padding, padding)
        end_point = (width - padding, height - padding)

        # Draw the rectangle
        img_with_rectangle = cv2.rectangle(
            img_copy, start_point, end_point, color, 5)

        # Put the label text on the image
        img_with_text = cv2.putText(img_with_rectangle, label, (50, height - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Save the modified image with a new name
        new_img_path = img_path.replace('.png', '_output.png')
        new_img_full_path = os.path.join('static', new_img_path)
        cv2.imwrite(new_img_full_path, img_with_text)

        # Append the prediction and new image path to the results
        results.append((label, new_img_path))

    return results

# Few Shot Learning Function to generate data using the Groq API and the given prompt
def generate_data(name, age, disease_name, clinical_history, symptoms, image_paths, diseases_level):
    img_url = url_for('static', filename=image_paths[0], _external=True)

    prompt = f"""
    I provided you the data of a patient and the image of the patient.
    The patient's name is {name}, age is {age}, disease name is {disease_name}, clinical history is {clinical_history}, symptoms are {symptoms}, and diseases level are {diseases_level}. The image is available at: {img_url}.
    Generate the data based on my given information. Note that the data should strictly be in the given dictionary format:
    {{
        'detailed findings': "Provide detailed findings based on the given data.",
        'clinical examination': "Provide clinical examination details based on the given data.",
        'imaging studies': ["Provide description", "Provide values finding"],
        'pathological staging': "Provide pathological staging details based on the given data.",
        'Recommended diet': ["Provide recommended diet 1", "Provide recommended diet 2", "Provide recommended diet 3"],
        'Recommended exercise': ["Provide recommended exercise 1", "Provide recommended exercise 2", "Provide recommended exercise 3"],
        'precautions': ["Provide precaution 1", "Provide precaution 2", "Provide precaution 3"]
    }}
    In your answer, only give the dictionary, nothing else. Just give the dictionary based on the given data.

    Example 1:
    {{
        'detailed findings': "A 45-year-old male patient presents with chest pain and shortness of breath.",
        'clinical examination': "Patient shows signs of respiratory distress and elevated heart rate.",
        'imaging studies': ["Chest X-ray shows enlarged heart", "CT scan reveals fluid in the lungs"],
        'pathological staging': "Stage II heart failure",
        'Recommended diet': ["Low sodium diet", "High fiber diet", "Increased water intake"],
        'Recommended exercise': ["Daily walking for 30 minutes", "Breathing exercises", "Light stretching"],
        'precautions': ["Avoid strenuous activities", "Monitor blood pressure regularly", "Follow up with cardiologist"]
    }}

    Example 2:
    {{
        'detailed findings': "A 30-year-old female patient presents with persistent headaches and dizziness.",
        'clinical examination': "Patient shows signs of elevated blood pressure and mild dehydration.",
        'imaging studies': ["MRI shows no abnormalities", "CT scan reveals slight sinus congestion"],
        'pathological staging': "No pathological staging required",
        'Recommended diet': ["Increase water intake", "Reduce caffeine consumption", "Balanced diet with fruits and vegetables"],
        'Recommended exercise': ["Daily walking for 20 minutes", "Yoga and relaxation exercises", "Light stretching"],
        'precautions': ["Avoid excessive caffeine", "Monitor blood pressure regularly", "Stay hydrated"]
    }}

    Example 3:
    {{
        'detailed findings': "A 60-year-old male patient presents with lower back pain and stiffness.",
        'clinical examination': "Patient shows signs of muscle tenderness and reduced range of motion in the lower back.",
        'imaging studies': ["X-ray shows mild degenerative changes in the lumbar spine", "MRI reveals slight disc bulge at L4-L5"],
        'pathological staging': "Stage I degenerative disc disease",
        'Recommended diet': ["Anti-inflammatory diet", "High fiber diet", "Increased water intake"],
        'Recommended exercise': ["Daily walking for 15 minutes", "Physical therapy exercises", "Core strengthening exercises"],
        'precautions': ["Avoid heavy lifting", "Practice good posture", "Follow up with physical therapist"]
    }}
    """

    # Replace with your Groq API Key
    groq_api_key = "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"
    # Replace with the actual API endpoint
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    messages = [
        {"role": "system", "content": "You are an AI assistant that strictly provides responses in dictionary format only, without any explanations or additional text. Note that some fields I didn't give to you but still you have to generate your own answer for those fields and always give the dictionary and not a single field should be empty in the dictionary."}
    ]
    # Adding user query
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "gemma2-9b-it",  # Using the specified model
        "messages": messages
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    data = response.json()['choices'][0]['message']['content']

    return data


@app.route('/chat-page')
def chat_page():
    return render_template('chat.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)