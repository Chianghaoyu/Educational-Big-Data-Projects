import os
import re
from flask import Flask, render_template, request, send_file, Blueprint, jsonify
from langchain_community.tools.tavily_search import TavilySearchResults
from arxiv import Search, SortCriterion

tavilySearch = Blueprint('tavilySearch', __name__)

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

os.environ['TAVILY_API_KEY'] = api_keys["TAVILY"]
# Initialize the TavilySearchResults tool
tool = TavilySearchResults(max_results=10)

def split_into_batches(text, batch_size=50):
    words = text.split()
    for i in range(0, len(words), batch_size):
        yield ' '.join(words[i:i + batch_size])


def search_papers_in_batches(keywords):
    all_papers = []
    batches = list(split_into_batches(keywords))
    
    batches = [f"site:arxiv.org search papers related to the following paragraph \"{batch}\"" for batch in batches]

    try:
        print(batches)
        response = tool.batch(batches)
        print(f"Response from Tavily API: {response}")
        
        for pack in response:
            print(pack)
            if isinstance(pack, list):
                papers = [{'url': result['url'], 'content': result['content']} for result in pack]
                all_papers.extend(papers)


    except Exception as e:
        print(f"Error occurred: {e}")
    
    
    arxiv_ids = [extract_arxiv_id(paper['url']) for paper in all_papers if extract_arxiv_id(paper['url'])]

    if arxiv_ids:
        return search_details(arxiv_ids)
    
    return all_papers


def search_details(arxiv_id_list):
    # Search arXiv for papers
    search = Search(id_list=arxiv_id_list,sort_by=SortCriterion.SubmittedDate)
    
    papers = []
    for result in search.results():

        paper_info = {
            "title": result.title,
            "abstract": result.summary,
            "pdf_url": result.pdf_url,
            "published": result.published.strftime('%Y-%m-%d'),
            "updated": result.updated.strftime('%Y-%m-%d')
        }

        papers.append(paper_info)
    
    print(papers)
    return papers

def extract_arxiv_id(url):
    pattern = r'arxiv.org/(?:abs|pdf|html)/([0-9]+\.[0-9]+[v0-9]*)'

    match = re.search(pattern, url)
    if match:
        print('match')
        return match.group(1)
    else:
        print(url)
    return None

@tavilySearch.route('/paraSearch', methods=['POST'])
def index():
    data = request.json
    query = data.get('query')
    papers = search_papers_in_batches(query)
    return jsonify(papers)

@tavilySearch.route('/download', methods=['POST'])
def download():
    pdf_url = request.form['pdf_url']
    response = request.get(pdf_url)
    
    # Extract the filename from the URL and save the PDF
    filename = pdf_url.split('/')[-1]
    pdf_path = os.path.join('downloads', filename)
    with open(pdf_path, 'wb') as f:
        f.write(response.content)

    return send_file(pdf_path, as_attachment=True)

