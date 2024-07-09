import os
import requests
from flask import Flask, render_template, request, send_file
from serpapi import GoogleSearch

app = Flask(__name__)
SERPAPI_API_KEY = '226288185e6fe224644cb4fd5b677530c82657eb364dd97870331b31656eb46e'  # Replace with your SerpApi API key

def search_papers(keywords):
    mySearch = GoogleSearch({
        "q": keywords,
        "engine": "google_scholar",
        "api_key": SERPAPI_API_KEY
    })

    results = mySearch.get_dict()
    papers = results.get('organic_results', [])
    return papers

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        keywords = request.form['keywords']
        papers = search_papers(keywords)
        return render_template('search.html', papers=papers, keywords=keywords)
    return render_template('search.html', papers=None)

@app.route('/download', methods=['POST'])
def download():
    pdf_url = request.form['pdf_url']
    response = requests.get(pdf_url)
    
    # Extract the filename from the URL and save the PDF
    filename = pdf_url.split('/')[-1]
    pdf_path = os.path.join('downloads', filename)
    with open(pdf_path, 'wb') as f:
        f.write(response.content)

    return send_file(pdf_path, as_attachment=True)

if __name__ == '__main__':
    if not os.path.exists('downloads'):
        os.makedirs('downloads')
    app.run(debug=True)
