import os

from groq import Groq
from langchain import *
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from data_creation import *
from langchain.chains import create_retrieval_chain

# Initializing the Groq client with an API key for LLM chat completion requests.

client = Groq(
    api_key = "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"
)

# Setting up the LLM (Large Language Model) using Google's Gemini model (version 1.5).
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=os.environ['GOOGLE_API_KEY'])

# Defining a prompt template that specifies the assistant's behavior and response guidelines.
# The prompt includes context about the dataset (Google stock trading data) and instructions on how to answer queries.

prompt = """
You are an intelligent assistant with access to historical stock trading data, specifically for Google (GOOGL). 
The dataset includes fields such as Open, High, Low, Close, Adjusted Close prices, Volume, and their corresponding timestamps (dates).

Your role is to help users by answering queries based on this dataset.

Inputs:
    - 'query': {input}
    - 'Context': {context}

Instructions:
1. For date-specific querY, return the exact value from the dataset (e.g., opening price, highest price, or trading volume on a given date).
2. For time range queries, such as weekly or monthly, compute and return the relevant statistics (e.g., highest or lowest price, average volume).
3. If the data is not available for a specific date or range, respond with: "Data not available for the requested period."
4. Respond concisely and factually, providing only the information requested without any extra commentary or speculation.
5. Ensure the response is clear, accurate, and formatted as a complete, grammatically correct sentence. Keep a conversational tone.

Answer to the query based solely on the provided Contxt and avoid making any assumptions.

ANSWER : 
"""

# Converting the string prompt to a ChatPromptTemplate format to be used in the LangChain framework.
prompt = ChatPromptTemplate.from_template(prompt)

# Creating a document chain using the LLM and the defined prompt template.
document_chain = create_stuff_documents_chain(llm, prompt)

# Initializing the retriever for accessing the database (db).
retriever = db.as_retriever()
docs = retriever.invoke("What was the volume on June 9, 2023?")
print(docs)

# Creating a retrieval chain by combining the document chain with the retriever.
# This setup allows for handling user queries by retrieving relevant data and generating responses based on the LLM.
retreival_chain = create_retrieval_chain(retriever, document_chain)

# Defining a sample user query to test the retrieval chain.
user_query = "What was the volume on June 9, 2023?"

# Invoking the retrieval chain with the user query to get the model's response.
response = retreival_chain.invoke({"input": user_query})

# Printing the final answer provided by the LLM for the user's query.
print(response['answer'])


# The following function, `get_llm_response`, is an alternate way to get responses using a different LLM model (Llama3).
# def get_llm_response(messages):
#     chat_completion = client.chat.completions.create(
#         messages= messages, 
#         temperature = 0, 
#         response_format={
#             "type": "json_object"
#         },

#         model="llama3-8b-8192",
#     )

#     return chat_completion.choices[0].message.content























































































































