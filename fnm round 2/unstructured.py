import os
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Set up the Google API Key
os.environ['GOOGLE_API_KEY'] = 'AIzaSyC7lzuoBxNhkkJKIda6JvGifTB-DfiCr04'

# Load the PDF file
loader = PyPDFLoader(file_path='trading101basics.pdf')  # Ensure the file path is correct
data = loader.load()

# Function to split the data into chunks
def chunk_data(data, chunk_size=50):
    """Splits data into chunks of specified row size or paragraph groupings."""
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

# Split the PDF data into chunks
chunked_data = chunk_data(data, chunk_size=100)

# Convert each chunk into a document
documents = [Document(page_content=str(chunk)) for chunk in chunked_data]

# Create embeddings using Google Generative AI
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create a FAISS vector store from the documents
db1 = FAISS.from_documents(documents, embeddings)

# Print the number of documents indexed
print(db1.index.ntotal)


