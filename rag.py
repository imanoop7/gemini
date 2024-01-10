import google.generativeai as genai
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


model=genai.GenerativeModel('gemini-pro')
embedding= GoogleGenerativeAIEmbeddings(model="models/embedding-001")
def get_response(question,context):
    prompt="""
    You are good assistant, will provide summary on {context} based on {question}
    """
    response= model.generate_content(prompt)
    return response.text

question = input()
new_db = FAISS.load_local("faiss", embedding)
res = new_db.similarity_search(question)


output = get_response(question, res[0])

print(output)



    


    