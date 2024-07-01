from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import gradio as gr
from PyPDF2 import PdfReader
import os

my_api_key = input("請輸入你的 API Key: ")
#os.environ["GOOGLE_API_KEY"] = my_api_key
os.environ["OPENAI_API_KEY"] = my_api_key
# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Load and preprocess PDF document
pdf_loader = PdfReader(r"C:\Users\a0981\OneDrive\桌面\Lectures\112-2\專題\paper\Interpretable Maching Learning with Brain Image and Survival Data\Interpretable Machine Learning with Brain Image and Survival Data.pdf")
raw_text = ''
for page in pdf_loader.pages:
    text = page.extract_text()
    if text:
        raw_text += text

# Initialize text splitter
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

# Initialize vector store (using FAISS)
vectorstore = FAISS.from_texts(texts, embeddings)

# Define prompt template
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say GTGTGTGTGTGTGTGTGTG, don't try to make up an answer.
{context}
Question: {question}
Helpful Answer:"""
QA_PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=['context', 'question']
)

# Initialize conversation retrieval chain
qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.8), vectorstore.as_retriever())

# Front-end web app setup using Gradio
with gr.Blocks() as demo:
    gr.Markdown("## Grounding DINO ChatBot")
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    chat_history = []

    def user(user_message, history):
        # Get response from QA chain
        response = qa({"question": user_message, "chat_history": history})
        # Append user message and response to chat history
        history.append((user_message, response["answer"]))
        return gr.update(value=""), history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

    if __name__ == "__main__":
        demo.launch()
