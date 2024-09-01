import os
import re
import openai
import requests
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from PyPDF2 import PdfReader
from io import BytesIO
from arxiv import Search, SortCriterion
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import google.generativeai as genai

from search_tavily import tavilySearch
app = Flask(__name__)
app.register_blueprint(tavilySearch)

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

# Load API keys
os.environ["OPENAI_API_KEY"] = api_keys["OpenAI"]
genai.configure(api_key=api_keys["GIMINI"])

# Initialize embeddings for the chatbot
embeddings = OpenAIEmbeddings()


# Initialize conversation retrieval chain
vectorstore = None
qa = None
def initialize_vectorstore_and_chain(pdf_path):
    global vectorstore, qa
    pdf_loader = PdfReader(pdf_path)
    raw_text = ''
    for page in pdf_loader.pages:
        text = page.extract_text()
        if text:
            raw_text += text

    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    vectorstore = FAISS.from_texts(texts, embeddings)
    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model="gpt-3.5-turbo-0125",temperature=0), vectorstore.as_retriever())

def summarize_text(text):
    if not text:
        return "No text provided for summarization", 400

    # Call the Google Generative AI to generate the summary
    question = "Can you make a summary of the article? "
    final_prompt = question + text
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    response = model.generate_content(final_prompt, stream=True)
    response.resolve()
    summary = response.text
    return summary

# Define the highlight function
def highlight_keywords(text, keywords):
    for keyword in keywords.split():
        text = re.sub(f'(?i)({keyword})', r'<mark>\1</mark>', text)
    return text


@app.route('/')
def home():
    return render_template('integrated_search2.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query')
    
    # Search arXiv for papers
    search = Search(
        query=query,
        max_results=100,
        sort_by=SortCriterion.SubmittedDate
    )
    
    papers = []
    for result in search.results():
        paper_info = {
            "title": result.title,
            "abstract": result.summary,
            "pdf_url": result.pdf_url,
            "published": result.published.strftime('%Y-%m-%d'),
            "updated": result.updated.strftime('%Y-%m-%d')
        }
        paper_info["highlighted_abstract"] = highlight_keywords(paper_info["abstract"], query)
        papers.append(paper_info)
    
    return jsonify(papers)

@app.route('/upload_file', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        pdf_path = os.path.join("uploads", file.filename)
        file.save(pdf_path)
        initialize_vectorstore_and_chain(pdf_path)
        return jsonify({'file_path': pdf_path})
    return jsonify({'error': 'No file uploaded'}), 400

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data.get('text')
    summary = summarize_text(text)
    return jsonify({'summary': summary})

@app.route('/download_and_summarize', methods=['POST'])
def download_and_summarize():
    data = request.json
    pdf_url = data.get('pdf_url')
    
    response = requests.get(pdf_url)
    pdf_bytes = BytesIO(response.content)
    pdf_reader = PdfReader(pdf_bytes)
    raw_text = ''
    for page in pdf_reader.pages:
        raw_text += page.extract_text()
    
    summary = summarize_text(raw_text)
    return jsonify({'summary': summary})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    context = data.get('context', '')
    
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"{context}\n\nQ: {message}\nA:",
        max_tokens=150
    )
    return jsonify({'response': response.choices[0].text.strip()})

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    user_message = data.get("message")
    chat_history = data.get("history", [])

    chat_history_tuples = [(user, bot) for user, bot in chat_history]
    response = qa({"question": user_message, "chat_history": chat_history_tuples})
    chat_history.append((user_message, response["answer"]))

    return jsonify(chat_history)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
