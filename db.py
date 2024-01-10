from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
def load_data(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    return pages

data = load_data("data.pdf")


db=FAISS.from_documents(data,embedding)
db.save_local('faiss')
