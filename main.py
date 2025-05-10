import requests
import json
import os
import cv2
import ast
import numpy as np
import logging
from typing import List, Tuple
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_cors import CORS
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, SelectField, TextAreaField, FileField, DateField
from wtforms.validators import DataRequired, Email, ValidationError, Length
from flask_wtf.file import FileField, FileRequired, FileAllowed
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from functools import lru_cache
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['PDF_FOLDER'] = "RAG Data"
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

db = SQLAlchemy(app)

# Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    contact_number = db.Column(db.String(15), nullable=False)
    date_of_birth = db.Column(db.Date, nullable=False)
    city = db.Column(db.String(50), nullable=False)
    user_type = db.Column(db.String(10), nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    age = db.Column(db.Integer, nullable=False)

# Forms
class RegisterForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired(), Length(min=2, max=50)])
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=8)])
    contact_number = StringField("Contact Number", validators=[DataRequired(), Length(min=10, max=15)])
    date_of_birth = DateField("Date of Birth", validators=[DataRequired()])
    city = StringField("City", validators=[DataRequired(), Length(min=2, max=50)])
    user_type = SelectField("User Type", choices=[('doctor', 'Doctor'), ('patient', 'Patient')], validators=[DataRequired()])
    gender = SelectField("Gender", choices=[('male', 'Male'), ('female', 'Female')], validators=[DataRequired()])
    submit = SubmitField("Register")

    def validate_email(self, field):
        if User.query.filter_by(email=field.data).first():
            raise ValidationError('Email already taken')

class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")

class DiagnosisForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired(), Length(min=2, max=50)])
    dob = DateField('Date of Birth', validators=[DataRequired()])
    disease_name = SelectField('Disease Name', choices=[('Breast Cancer', 'Breast Cancer')], validators=[DataRequired()])
    ct_images = FileField('Upload CT Images', validators=[FileRequired(), FileAllowed(['png', 'jpg', 'jpeg'], 'Images only!')])
    clinical_history = TextAreaField('Clinical History', validators=[DataRequired(), Length(min=10)])
    symptoms = StringField('Symptoms', validators=[DataRequired(), Length(min=5)])
    submit = SubmitField('Generate Report')

    def validate_name(self, field):
        if not field.data.replace(' ', '').isalpha():
            raise ValidationError('Name must contain only letters')

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PDF_FOLDER'], exist_ok=True)

# TensorFlow Model Singleton
_model = None
def get_model():
    global _model
    if _model is None:
        logger.warning("Model was not loaded at startup; using fallback.")
    return _model

# RAG Document Processor
PERSONAL_REPORTS = []

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self.update_vector_store([])

    def extract_text_from_pdfs(self, folder_path: str) -> List[str]:
        try:
            extracted_texts = []
            for pdf_file in os.listdir(folder_path):
                if pdf_file.lower().endswith(".pdf"):
                    file_path = os.path.join(folder_path, pdf_file)
                    try:
                        with fitz.open(file_path) as doc:
                            text = "".join(page.get_text("text") + "\n" for page in doc)
                            extracted_texts.append(text)
                    except Exception as e:
                        logger.error(f"Error processing {pdf_file}: {str(e)}")
            return extracted_texts
        except Exception as e:
            logger.error(f"Error in PDF extraction: {str(e)}")
            return []

    def update_vector_store(self, current_report: List[str]):  # Add parameter
            try:
                pdf_texts = self.extract_text_from_pdfs(app.config['PDF_FOLDER'])
                all_documents = current_report + pdf_texts  # Use the passed current_report
                if all_documents:
                    docs = self.text_splitter.create_documents(all_documents)
                    self.vector_store = FAISS.from_documents(docs, self.embedding_model)
                else:
                    self.vector_store = None
                logger.info("Vector store updated successfully with current report")
            except Exception as e:
                logger.error(f"Error updating vector store: {str(e)}")
                raise


    def retrieve_context(self, query: str, k: int = 3) -> str:
        if not self.vector_store:
            return "No context available yet. Please generate a report first."
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return "\n".join(doc.page_content for doc in results)
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return ""

# Groq Client
class GroqClient:
    @staticmethod
    @lru_cache(maxsize=50)
    def call_groq_api(prompt: str, model: str = "gemma2-9b-it") -> str:
        if not GROQ_API_KEY:
            logger.error("GROQ_API_KEY not configured")
            return "API key not configured"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        data = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "max_tokens": 500}
        try:
            response = requests.post(GROQ_API_URL, json=data, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API request failed: {str(e)}")
            return "Sorry, I couldn't process that request."

# Global Document Processor
doc_processor = None

@lru_cache(maxsize=100)
def model_predict(image_paths: tuple) -> List[Tuple[str, str]]:
    model = get_model()
    results = []
    for img_path in image_paths:
        img_full_path = os.path.join('static', img_path)
        if not os.path.exists(img_full_path):
            logger.error(f"File not found: {img_full_path}")
            continue
        img_cv = cv2.imread(img_full_path)
        if img_cv is None:
            logger.error(f"Failed to load image: {img_full_path}")
            continue
        img_copy = img_cv.copy()
        img_resized = cv2.resize(img_cv, (224, 224))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        if model is None:
            # Fallback prediction if model failed to load
            logger.warning(f"Model unavailable, using default prediction for {img_path}")
            label = "Cancer: Unknown (Model Unavailable)"
            color = (255, 255, 0)  # Yellow for unknown
        else:
            try:
                prediction = model.predict(img_array, verbose=0)
                label = "Cancer: Yes (Malignant)" if prediction[0][0] >= 0.5 else "Cancer: No (Non-Malignant)"
                color = (0, 0, 255) if prediction[0][0] >= 0.5 else (0, 255, 0)
            except Exception as e:
                logger.error(f"Prediction failed for {img_path}: {str(e)}")
                label = "Cancer: Unknown (Prediction Error)"
                color = (255, 255, 0)

        height, width, _ = img_copy.shape
        padding = 100
        start_point = (padding, padding)
        end_point = (width - padding, height - padding)
        img_with_rectangle = cv2.rectangle(img_copy, start_point, end_point, color, 5)
        img_with_text = cv2.putText(img_with_rectangle, label, (50, height - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        new_img_path = img_path.rsplit('.', 1)[0] + '_output.png'
        new_img_full_path = os.path.join('static', new_img_path)
        cv2.imwrite(new_img_full_path, img_with_text)
        results.append((label, new_img_path))
    return results

@lru_cache(maxsize=50)
def generate_data(name: str, age: int, disease_name: str, clinical_history: str, 
                 symptoms: tuple, image_paths: tuple, diseases_level: tuple) -> dict:
    if not image_paths:
        logger.warning("No image paths provided")
        return {}
    img_url = url_for('static', filename=image_paths[0], _external=True)
    prompt = (
        "I provided you the data of a patient and the image of the patient.\n\n"
        f"The patient's name is {name}, age is {age}, disease name is {disease_name}, "
        f"clinical history is {clinical_history}, symptoms are {list(symptoms)}, "
        f"and diseases level are {list(diseases_level)}. The image is available at: {img_url}.\n\n"
        "Generate the data based on my given information. Note that the data should strictly be in the given dictionary format:\n"
        "{\n"
        "    'detailed findings': 'Provide detailed findings based on the given data.',\n"
        "    'clinical examination': 'Provide clinical examination details based on the given data.',\n"
        "    'imaging studies': ['Provide description', 'Provide values finding'],\n"
        "    'pathological staging': 'Provide pathological staging details based on the given data.',\n"
        "    'Recommended diet': ['Provide recommended diet 1', 'Provide recommended diet 2', 'Provide recommended diet 3'],\n"
        "    'Recommended exercise': ['Provide recommended exercise 1', 'Provide recommended exercise 2', 'Provide recommended exercise 3'],\n"
        "    'precautions': ['Provide precaution 1', 'Provide precaution 2', 'Provide precaution 3']\n"
        "}\n"
        "In your answer, only give the dictionary, nothing else."
    )
    messages = [
        {"role": "system", "content": "You are an AI assistant that strictly provides responses in dictionary format only, without any explanations or additional text."},
        {"role": "user", "content": prompt}
    ]
    return ast.literal_eval(GroqClient.call_groq_api(prompt))

def chat_with_bot(user_input: str) -> str:
    if not user_input.strip():
        return "Please enter a valid query."
    context = doc_processor.retrieve_context(user_input)
    prompt = f"Use the following context to answer the question.\nContext:\n{context}\nUser Question: {user_input}\nAnswer:"
    return GroqClient.call_groq_api(prompt, model="gemma2-9b-it")

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        name = form.name.data
        email = form.email.data
        password = form.password.data.encode('utf-8')
        contact_number = form.contact_number.data
        date_of_birth = form.date_of_birth.data
        city = form.city.data
        user_type = form.user_type.data
        gender = form.gender.data
        age = datetime.now().year - date_of_birth.year - ((datetime.now().month, datetime.now().day) < (date_of_birth.month, date_of_birth.day))
        hashed_password = bcrypt.hashpw(password, bcrypt.gensalt())
        new_user = User(name=name, email=email, password=hashed_password, contact_number=contact_number,
                       date_of_birth=date_of_birth, city=city, user_type=user_type, gender=gender, age=age)
        try:
            db.session.add(new_user)
            db.session.commit()
            flash("Registration successful! Please login.")
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Database error during registration: {str(e)}")
            flash("Registration failed. Please try again.")
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data.encode('utf-8')
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.checkpw(password, user.password):
            session['user_id'] = user.id
            session['name'] = user.name
            session['user_type'] = user.user_type
            return redirect(url_for('form'))
        flash("Login failed. Please check your email and password.")
    return render_template('login.html', form=form)

@app.route('/form', methods=['GET', 'POST'])
def form():
    form = DiagnosisForm()
    return render_template('form.html', form=form)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('name', None)
    session.pop('user_type', None)
    flash("You have been logged out successfully.")
    return redirect(url_for('login'))

@app.route('/generate_report', methods=['POST'])
def generate_report():
    if 'user_id' not in session:
        flash("Please login first.")
        return redirect(url_for('login'))
    try:
        name = request.form['name']
        dob = datetime.strptime(request.form['dob'], '%Y-%m-%d').date()
        disease_name = request.form['disease_name']
        clinical_history = request.form['clinical_history']
        symptoms_json = request.form.get('symptoms_json', '[]')
        symptoms = tuple(json.loads(symptoms_json)) if symptoms_json else tuple()
        prepared_by = session.get('name')
        age = datetime.now().year - dob.year - ((datetime.now().month, datetime.now().day) < (dob.month, dob.day))
        ct_images = request.files.getlist('ct_images')
        image_paths = []
        for image in ct_images:
            if image and image.filename:
                filename = image.filename
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(image_path)
                relative_path = f'uploads/{filename}'
                image_paths.append(relative_path)
        image_paths_tuple = tuple(image_paths)
        results = model_predict(image_paths_tuple)
        diseases_level = tuple(result[0] for result in results)
        new_image_paths = tuple(result[1] for result in results)
        data = generate_data(name, age, disease_name, clinical_history, symptoms, new_image_paths, diseases_level)
        data = ast.literal_eval(data) if isinstance(data, str) else data
        current_date = datetime.now().strftime('%Y-%m-%d')
        user_type = session.get('user_type')
        
        
        # Append new report details to PERSONAL_REPORTS
        PERSONAL_REPORTS.clear()
        new_report = (
            f"Patient Name: {name}\n"
            f"Date of Birth: {dob}\n"
            f"Age: {age}\n"
            f"Date of Report: {current_date}\n"
            f"Disease Name: {disease_name}\n"
            f"Disease Level: {', '.join(diseases_level)}\n"
            f"Clinical History: {clinical_history}\n"
            f"Symptoms: {', '.join(symptoms)}\n"
            f"Clinical Examination: {data.get('clinical examination', 'N/A')}\n"
            f"Imaging Studies: {', '.join(data.get('imaging studies', ['N/A']))}\n"
            f"Pathological Staging: {data.get('pathological staging', 'N/A')}\n"
            f"Precautions: {', '.join(data.get('precautions', ['N/A']))}\n"
            f"Recommended Diet: {', '.join(data.get('Recommended diet', ['N/A']))}\n"
            f"Recommended Exercise: {', '.join(data.get('Recommended exercise', ['N/A']))}\n"
            f"Prepared By: {prepared_by}\n"
        )
        PERSONAL_REPORTS.append(new_report)
        doc_processor.update_vector_store(PERSONAL_REPORTS)

        return render_template('report.html', user_type=user_type, name=name, dob=dob, age=age, 
                             disease_name=disease_name, clinical_history=clinical_history, 
                             symptoms=list(symptoms), prepared_by=prepared_by, image_paths=list(new_image_paths), 
                             diseases_level=list(diseases_level), data=data, current_date=current_date)
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        flash("An error occurred while generating the report. Please try again.")
        return redirect(url_for('form'))

@app.route('/chat-page')
def chat_page():
    return render_template('chat.html')

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        user_query = data.get("query", "").strip()
        if not user_query:
            return jsonify({"response": "Please enter a valid query."}), 400
        response = chat_with_bot(user_query)
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"response": "An error occurred while processing your request."}), 500

def init_app():
    with app.app_context():
        global doc_processor, _model
        db.create_all()
        doc_processor = DocumentProcessor()
        doc_processor.update_vector_store([])
        # Load model at startup
        try:
            logger.info(f"TensorFlow version: {tf.__version__}")
            logger.info(f"Loading model from: {os.path.abspath('D:/Project/Mini Project 2/Modal/breast_cancer.keras')}")
            _model = load_model('D:/Project/Mini Project 2/Modal/breast_cancer.keras', compile=False)
            logger.info("Model loaded successfully")
            logger.info(f"Model summary: {_model.summary()}")
        except Exception as e:
            logger.error(f"Failed to load model at startup: {str(e)}. Using fallback.")
            _model = None
    logger.info("Application initialized successfully")

if __name__ == '__main__':
    init_app()
    app.run(debug=False)