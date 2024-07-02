import os
from flask import Flask, request, render_template, send_from_directory, jsonify
from PyPDF2 import PdfReader
from transformers import pipeline
import google.generativeai as genai
#from dotenv import load_dotenv

# Initialize the summarizer pipeline
#summarizer = pipeline("summarization")
# Load environment variables
#load_dotenv()


# Function to read API keys from file
def read_api_keys(file_path):
    api_keys = {}
    with open(file_path, "r") as file:
        for line in file:
            key, value = line.strip().split('=')
            api_keys[key] = value
    return api_keys

# Read API keys from file
api_keys = read_api_keys(r"C:\Users\a0981\OneDrive\桌面\api_key.txt")

# Set API keys as environment variables
my_api_key = api_keys["GIMINI"]
# Configure Google Generative AI with the API key
genai.configure(api_key=my_api_key)

app = Flask(__name__)
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# Ensure the 'uploads' directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        return jsonify({'file_path': file_path})
    return "No file uploaded", 400

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/extract_text', methods=['POST'])
def extract_text():
    pdf_path = request.json['pdf_path']
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text() + "\n"
        return jsonify({'text': text})
    except Exception as e:
        return str(e), 500

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.json['text']
    if not text:
        return "No text provided for summarization", 400

    # Call the Google Generative AI to generate the summary
    question = "Can you make a summary of the article? "
    final_prompt = question + text
    response = model.generate_content(final_prompt, stream=True)
    response.resolve()
    summary = response.text
    #summary = response["choices"][0]["text"].strip()

    #summary = summarizer(text, do_sample=False)[0]['summary_text']
    return jsonify({'summary': summary})

@app.route('/save_selected_text', methods=['POST'])
def save_selected_text():
    selected_text = request.json.get('selectedText')
    with open('selected_text.txt', 'w') as f:
        f.write(selected_text)
    return '', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
