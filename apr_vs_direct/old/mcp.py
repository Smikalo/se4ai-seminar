from langchain.agents import Tool
from langchain_openai import ChatOpenAI

from pdf import mcp_database, pdf_texts
from langchain.tools import Tool
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

def dummy_mcp_query(query: str) -> str:
    """Simulate an MCP server query by looking up a key in the structured database."""
    # For simplicity, do a case-insensitive lookup for a known key substring
    for key, value in mcp_database.items():
        if key.lower() in query.lower():
            return value
    # If nothing found, return a default message
    return "MCP: No direct data available."

def dummy_rag_search(query: str) -> str:
    """Simulate a RAG search over PDF texts by simple keyword matching."""
    query_lower = query.lower()
    results = []
    for doc in pdf_texts:
        # Split doc into sentences for granularity
        for sentence in doc.split(". "):
            if query_lower in sentence.lower():
                results.append(sentence.strip())
    if not results:
        return "RAG: No relevant information found in documents."
    # Return the most relevant sentence (here, just the first match for demo)
    return "RAG: " + results[0]

def mcp_query(query: str) -> str:
    for key, value in mcp_database.items():
        if query.lower() in key.lower():
            return value
    return "No direct data available."
# Wrap these functions as LangChain tools
mcp_tool = Tool(
    name="MCP_Database",
    func=mcp_query,
    description="Access company structured data, such as error codes or policy thresholds."
)

documents = [Document(page_content=text) for text in pdf_texts]
embedding = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embedding)
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-4"), retriever=retriever)

rag_tool = Tool(
    name="RAG_Search",
    func=lambda q: qa_chain.run(q),
    description="Useful for answering questions from company documentation (PDFs)."
)
