import os
import requests
from flask import Flask, render_template, request, redirect, url_for, send_file
from io import BytesIO
import re

app = Flask(__name__)

# Function to search for papers on arXiv, sorted by date
def search_arxiv(query, max_results=10):
    base_url = "http://export.arxiv.org/api/query?"
    search_query = f"search_query=all:{query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    response = requests.get(base_url + search_query)
    return response.text

# Function to parse the response and extract relevant information
def parse_arxiv_response(response, query):
    import xml.etree.ElementTree as ET
    root = ET.fromstring(response)
    papers = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        paper_id = entry.find("{http://www.w3.org/2005/Atom}id").text.split("/")[-1]
        title = entry.find("{http://www.w3.org/2005/Atom}title").text
        summary = entry.find("{http://www.w3.org/2005/Atom}summary").text

        # Highlight keywords in the summary
        keywords = query.split()
        for keyword in keywords:
            summary = re.sub(f"(?i)({keyword})", r'<span class="highlight">\1</span>', summary)

        pdf_url = entry.find("{http://www.w3.org/2005/Atom}link[@title='pdf']").attrib['href']
        papers.append({
            "id": paper_id,
            "title": title,
            "summary": summary,
            "pdf_url": pdf_url
        })
    return papers

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["query"]
        response = search_arxiv(query)
        papers = parse_arxiv_response(response, query)
        return render_template("search_arxiv.html", papers=papers, query=query)
    return render_template("search_arxiv.html", papers=[], query="")

@app.route("/download/<paper_id>")
def download_paper(paper_id):
    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    response = requests.get(pdf_url)
    return send_file(BytesIO(response.content), download_name=f"{paper_id}.pdf")

if __name__ == "__main__":
    app.run(debug=True)
