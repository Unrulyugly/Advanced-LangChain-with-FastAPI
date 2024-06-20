import requests
import pandas as pd
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


CONNECTION_STRING = "postgresql://langchain:langchain@localhost:6024/langchain"
COLLECTION_NAME= "langchain_pg_embeddding"
embeddings =NomicEmbeddings(model="nomic-embed-text-v1")
file_path = 'google_engine/email_domain_results.csv'

# Read the CSV file using pandas
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Load the data into documents using DataFrameLoader
loader = DataFrameLoader(df, page_content_column='Snippet')
docs = loader.load()

# Split text into chunks of 512 tokens, with 20% token overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=612,
    chunk_overlap=103,
)
documents = text_splitter.split_documents(docs)

# Convert the documents into a list of dictionaries
docs_data = [doc.dict() for doc in documents]

# Prepare the data to be sent in the POST request
url = "http://localhost:8000/index?cleanup=full"
response = requests.post(url, json=docs_data)

# Print the response text for debugging
print(response.text)

try:
    # Print the JSON response from the server
    print(response.json())
except requests.exceptions.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
