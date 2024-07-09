import os
from flask import Flask, render_template, request, send_file
from langchain_community.tools.tavily_search import TavilySearchResults

app = Flask(__name__)

os.environ['TAVILY_API_KEY'] = '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'
# Initialize the TavilySearchResults tool
tool = TavilySearchResults(max_results=10)

def search_papers(keywords):
    try:
        keywords = f"search {keywords} in google scholar"
        print(f"Sending query to Tavily API: {keywords}")
        response = tool.invoke(keywords)
        print(f"Response from Tavily API: {response}")
        
        # Handle the response format
        if isinstance(response, list):
            papers = [{'url': result['url'], 'content': result['content']} for result in response]
        else:
            papers = []
        
    except request.HTTPError as e:
        print(f"HTTP Error occurred: {e}")
        papers = []
    except Exception as e:
        print(f"Error occurred: {e}")
        papers = []
    
    return papers

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        keywords = request.form['keywords']
        papers = search_papers(keywords)
        return render_template('search_tavily.html', papers=papers, keywords=keywords)
    return render_template('search_tavily.html', papers=None)

@app.route('/download', methods=['POST'])
def download():
    pdf_url = request.form['pdf_url']
    response = request.get(pdf_url)
    
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