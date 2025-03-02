from flask import Flask, request, render_template, send_file
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import google.generativeai as genai
from fpdf import FPDF
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__, template_folder="web/templates")

# Load Google Generative AI Key
genai.configure(api_key="AIzaSyBl_u8dpgKuDaeTmWad87sCr2JUiEVkACA")  # Replace with your actual API Key

# Load the trained model
model = load_model("brain_tumor_classifier.h5")

# Define tumor classes
TUMOR_CLASSES = ["No Tumor", "Glioma", "Meningioma", "Pituitary Tumor"]

# Function to preprocess an image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None  # Handle invalid image
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to get tumor information from Google Generative AI
def get_tumor_info(tumor_type):
    prompt = f"What is {tumor_type} and what are the precautionary steps to follow?"
    model = genai.GenerativeModel("gemini-1.5-pro")
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error fetching tumor information: {e}"

# Function to generate PDF report
def generate_pdf(filename, prediction, confidence, tumor_info):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(0, 10, "Brain Tumor Classification Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, f"Patient Image: {filename}", ln=True)
    pdf.cell(0, 10, f"Predicted Tumor Type: {prediction}", ln=True)
    pdf.cell(0, 10, f"Confidence: {confidence:.2f}", ln=True)
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    
    # Properly format bold text and bullet points with word wrapping
    for line in tumor_info.split('\n'):
        line = line.strip()
        if not line:
            pdf.ln(5)
            continue

        if line.startswith("*"):
            pdf.set_font("Arial", style="B", size=12)
            pdf.cell(5)
            pdf.multi_cell(0, 10, line.replace("*", ""))
            pdf.set_font("Arial", size=12)
        elif "**" in line:
            parts = line.split("**")
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    pdf.set_font("Arial", style="B", size=12)
                    pdf.multi_cell(0, 10, part)
                    pdf.set_font("Arial", size=12)
                else:
                    pdf.multi_cell(0, 10, part)
        else:
            pdf.multi_cell(0, 10, line)

    report_filename = f"reports/{filename}.pdf"
    pdf.output(report_filename)
    return report_filename

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Secure filename and save uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join("uploads", filename)
    file.save(file_path)

    # Preprocess and predict
    processed_image = preprocess_image(file_path)
    if processed_image is None:
        os.remove(file_path)  # Remove the invalid file
        return "Invalid image file", 400

    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class = TUMOR_CLASSES[predicted_class_index]
    confidence = float(np.max(predictions))

    # Get tumor information from Google AI
    tumor_info = get_tumor_info(predicted_class)

    # Generate PDF report
    os.makedirs("reports", exist_ok=True)
    pdf_path = generate_pdf(filename, predicted_class, confidence, tumor_info)

    # Remove uploaded file after processing
    if os.path.exists(file_path):
        os.remove(file_path)

    # Directly send the PDF report for download
    return send_file(pdf_path, as_attachment=True)

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    app.run(debug=True, port=5000)
