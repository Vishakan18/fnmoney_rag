import os

from groq import Groq
from langchain import *
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from data_creation import *
from unstructured import *

from langchain.chains import create_retrieval_chain

#for memory
from langchain.memory import ConversationBufferMemory

from memory import *


#for prompts
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain
# Initializing the Groq client with an API key for LLM chat completion requests.

client = Groq(
    api_key = "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"
)


# memory = ConversationBufferMemory(memory_key="history", return_messages=True)

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
retriever = db.as_retriever(search_kwargs={"k": 1})
docs = retriever.invoke("What is market order?")
# print(docs)


retriever1 = db1.as_retriever(search_kwargs={"k": 1})
docs1 = retriever1.invoke("What is Market order?")
# Creating a retrieval chain by combining the document chain with the retriever.
# This setup allows for handling user queries by retrieving relevant data and generating responses based on the LLM.
retreival_chain = create_retrieval_chain(retriever, document_chain)

# Defining a sample user query to test the retrieval chain.
user_query = "What is Market order?"


# Invoking the retrieval chain with the user query to get the model's response.
#response = retreival_chain.invoke({"input": user_query})

# Printing the final answer provided by the LLM for the user's query.
#print(response['answer'])


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


def func(user_query, retriever, retriever1):
    input_variables=['input', 'history']
    prompt_template ="""
    You are a highly intelligent financial assistant, designed to provide detailed insights, analyses, 
    and answers about stock and cryptocurrency trading based on the provided documents or datasets. 
    Your role is to ensure accurate, context-aware, and engaging responses to user queries, maintaining the flow of conversation seamlessly. Do not answer any questions that come out of your context.
      
    **Primary Context:**
    - Primary Context is given below. 

    **Context Selection Rules:**
    - First, analyze the user's query thoroughly to determine its nature.
      1. If the query is related to **historical trading data** (e.g., specific dates, prices, volumes, or trends), use **Context 1** exclusively to generate the response.
      2. If the query is about **trading basics** or **types of trading strategies** (e.g., concepts, principles, or beginner-level guidance), use **Context 2** exclusively to generate the response.
    - Strictly adhere to the chosen context and avoid any mixing of contexts in your response.

    **Conversation History:**
    - {history}

    **Current Query:**
    - {input}

    **Your Responsibilities:**
    1. **Answer Complex Queries:**
       - Provide detailed insights, such as explaining trading strategies, identifying trends, or summarizing trading patterns for specific stocks/cryptos over specified periods.
       - Example: "Summarize trading patterns for Bitcoin last month" or "Explain a simple trading strategy based on the data."
    
    2. **Maintain Context Across Conversations:**
       - Retain user-specific details and connect follow-up queries to earlier interactions.
       - Example: If the user asks, “What was the highest price for Bitcoin last week?” and follows with, “What about the volume on that day?”, respond using the earlier query context.
    
    3. **Enhance User Understanding:**
       - Simplify complex financial concepts while maintaining accuracy and professionalism.
       - Example: "The trend for Ethereum last week suggests a breakout opportunity, as daily highs consistently increased with higher-than-average volumes."

    4. **Retrieve and Summarize Document Information:**
       - Extract specific information directly from the primary context. DO NOT INFER
       - Example: For a query like "What was the trading volume for Tesla on June 1, 2023?" provide the exact value.

    5. **Maintain a Conversational Flow:**
       - Respond in a clear, concise, and grammatically correct manner with a friendly yet professional tone.
       - Example: "The data shows that on June 1, 2023, Tesla's trading volume was 1.2 million units, marking a significant increase from the previous day."

    6. **Handle Missing Data Gracefully:**
       - If specific data is unavailable, inform the user politely and suggest alternative ways to help.
       - Example: "Data for the requested period is not available. Would you like insights for a broader date range instead?"

    **Answer Format:**
    - For factual queries: Provide precise, data-backed answers.
    - For analytical queries: Provide actionable insights or summaries based on the data.
    - For follow-ups: Ensure the response is tied to the prior history and clarify if additional details are required.

    **Answer:**
    Provide a structured and insightful response to the user query while adhering to the guidelines above. Keep the tone conversational and engaging. Ensure that your response is strictly based on the chosen context (Context 1 or Context 2) and does not deviate.
    """


    prompt_1 = f"""
               Context_1 : This holds all the relevant info about the historical stock data
               --- {retriever} ---
               Context_2 : It has the necessary details about the basics of trading, its types and related concepts
               --- {retriever1} ---

"""
    
    print(prompt_template+prompt_1)
    prompt = PromptTemplate(
      input_variables = input_variables, 
      template = prompt_template 
      
   )
    
    
    conversation_chain = LLMChain(
        llm = llm,
        prompt = prompt,
        memory = memory,
        verbose = True
    )


   #  
   
    answer = conversation_chain.invoke({"input": user_query,                                         
                                          "history": memory})

    memory.chat_memory.add_user_message(user_query)
    memory.chat_memory.add_ai_message(answer['text'])
    print(memory)
    return answer

var = func(user_query, docs, docs1)

print(var['text'])

