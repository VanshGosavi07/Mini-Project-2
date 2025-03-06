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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


# Function to Navigate home page
@app.route('/')
def home():
    return render_template('home.html')

# Function to Generate the Report from form Data
@app.route('/generate_report', methods=['POST'])
def generate_report():
    if request.method == 'POST':
        # Gather form data
        name = request.form['name']
        dob = request.form['dob']
        disease_name = request.form['disease_name']
        clinical_history = request.form['clinical_history']
        prepared_by = request.form['prepared_by']

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
        return render_template('report.html', name=name, dob=dob, age=age, disease_name=disease_name, clinical_history=clinical_history, symptoms=symptoms, prepared_by=prepared_by, image_paths=new_image_paths, diseases_level=diseases_level, data=data, current_date=current_date)

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
    app.run(debug=True)