import yfinance as yf
import pandas as pd
import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

os.environ['GOOGLE_API_KEY'] = 'AIzaSyC7lzuoBxNhkkJKIda6JvGifTB-DfiCr04'


# # Define the stock symbol for Google (Alphabet Inc.)
# stock_symbol = "GOOGL"  # Symbol for Alphabet Inc. (Google)
# start_date = "2020-01-01"
# end_date = "2024-11-01"

# # Fetch Google stock data
# google_stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# # Display the first few rows of the data
# print(google_stock_data.head())

# # Save the stock data to a CSV file
# csv_file_path = "google_stock_data.csv"
# google_stock_data.to_csv(csv_file_path)

# print(f"Google stock data saved to {csv_file_path}")


loader = CSVLoader(file_path='google_stock_data.csv', encoding='utf-8', csv_args={'delimiter': ','})
data = loader.load()

def chunk_data(data, chunk_size=50):
    """Splits data into chunks of specified row size."""
    return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

# Split the data into chunks of 50 rows each
chunked_data = chunk_data(data, chunk_size=100)

# Each chunk can now be treated as a separate document
documents = [Document(page_content=str(chunk)) for chunk in chunked_data]
    # documents = loader.load()
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.from_documents(documents, embeddings)
print(db.index.ntotal)



