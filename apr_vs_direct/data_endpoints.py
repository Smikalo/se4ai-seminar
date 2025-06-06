# data_endpoints.py
from flask import Flask, request, jsonify
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
DOCS_DIRECTORY = "business_docs"
os.makedirs(DOCS_DIRECTORY, exist_ok=True)

# Create dummy .txt files
hr_policy_content = "This document contains the Human Resources policies of the organization. The vacation policy allows for 20 days of paid leave per year for all full-time employees. Sick leave is 10 days per year."
finance_report_content = "This is the financial report for the year 2023. The total revenue was $5 million. The net profit was $1.2 million."

with open(os.path.join(DOCS_DIRECTORY, "hr_policy.txt"), "w") as f:
    f.write(hr_policy_content)

with open(os.path.join(DOCS_DIRECTORY, "finance_report_2023.txt"), "w") as f:
    f.write(finance_report_content)

# --- Data Loading ---
loader = DirectoryLoader(DOCS_DIRECTORY, glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)

# --- Flask App ---
app = Flask(__name__)

@app.route('/rag/search', methods=['POST'])
def rag_search():
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "Query not provided"}), 400
    results = db.similarity_search(query)
    return jsonify([{"content": doc.page_content, "source": doc.metadata.get('source')} for doc in results])

@app.route('/mcp/metadata', methods=['GET'])
def mcp_metadata():
    metadata = [doc.metadata for doc in documents]
    return jsonify(metadata)

if __name__ == '__main__':
    app.run(port=5000)
    #podcasts linkedin techranch touchpoints with people talk a lot step by s
    # technical assets maher ask, steffen 50