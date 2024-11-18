import streamlit as st
import pandas as pd
import requests
from langchain_ollama import OllamaLLM
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.tools import tool
from langchain.tools.render import render_text_description
from langchain_core.output_parsers import JsonOutputParser
from operator import itemgetter
from langchain_core.messages import HumanMessage, AIMessage
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build

x="qwen2.5:3b-instruct"
st.toast=("hello")
model = Ollama(model=x)

chat_history = [] # Store the chat history

@tool
def converse(input: str) -> str:
    """Provide a natural language response using the user input."""
    bar.progress(40)
    return model.invoke(input)

#tools = [repl, converse ,recognize_speech_from_microphone , ]
tools = [
    converse
]


# Configure the system prompts
rendered_tools = render_text_description(tools)

system_prompt = f"""You answer questions with simple answers and no funny stuff , You are an assistant that has access to the following set of tools. Here are the names and descriptions for each tool:

{rendered_tools}

Given the user input, return the name and input of the tool to use. Return your response as a JSON blob with 'name' and 'arguments' keys. The value associated with the 'arguments' key should be a dictionary of parameters."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
     MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ]
)

# Define a function which returns the chosen tools as a runnable, based on user input.
def tool_chain(model_output):
    tool_map = {tool.name: tool for tool in tools}
    chosen_tool = tool_map[model_output["name"]]
    return itemgetter("arguments") | chosen_tool

# The main chain: an LLM with tools.
chain = prompt | model | JsonOutputParser() | tool_chain





# Set up message history.
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("From calculations to image generation, data analysis to task prioritization, I'm here to assist. Always on, always learning. How can I help you today?")

# Set the page title.
st.title("Breakout Ai")

# Render the chat history.
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# React to user input
if input := st.chat_input("What is up?"):

    if input == "/clear":
        #print("Chat history cleared.")
        st.chat_message("assistant").write("Chat history cleared.")
        st.toast("Data Cleared")

    else:
        # Display user input and save to message history.
        st.chat_message("user").write(input)
        msgs.add_user_message(input)

        # Invoke chain to get response.
        bar = st.progress(0)
        response = chain.invoke({"input": input, "chat_history": chat_history})
        chat_history.append(HumanMessage(content=input))
        chat_history.append(AIMessage(content=response))
        bar.progress(90)

        # Display AI assistant response and save to message history.
        st.chat_message("assistant").write(str(response))
        msgs.add_ai_message(response)

        bar.progress(100)

        # Ensure the model retains context
        #msgs.add_ai_message(model.invoke(input))

# Function to authenticate and fetch data from Google Sheets
def fetch_google_sheet(sheet_url):
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
    )
    sheet_id = sheet_url.split("/")[5]
    service = build("sheets", "v4", credentials=credentials)
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=sheet_id, range="Sheet1").execute()
    return pd.DataFrame(result.get("values", [])[1:], columns=result.get("values", [])[0])


# Web search using SerpAPI
def perform_web_search(query):
    api_key ="c86d41704a25785b2c369e9447a78449447e86943e51820152c8d3916f21aa6f"  # Store your API key in secrets.toml
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
    }
    response = requests.get("https://serpapi.com/search", params=params)
    return response.json()


# Process search results with Ollama or OpenAI
def extract_information_with_llm(search_results, prompt_template):
    combined_results = "\n".join([result["snippet"] for result in search_results.get("organic_results", [])])
    prompt = prompt_template.format(results=combined_results)

    # Call Ollama or OpenAI API
    llm_endpoint = "http://localhost:11434/api/v1/generate"  # Ollama local URL
    response = requests.post(
        llm_endpoint,
        json={"model": "llama", "prompt": prompt},
    )
    if response.status_code == 200:
        return response.json().get("response", "").strip()
    else:
        st.error("Error: Unable to process with LLM API")
        return "Error in processing"

def query_ollama(prompt):
    url = "http://localhost:11411/v1/query"  # This is the default Ollama API endpoint
    headers = {
        'Content-Type': 'application/json',
    }
    data = {
        "model": "llama2",  # specify the model (you can change it depending on what you need)
        "input": prompt
    }

    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error with Ollama API: {response.text}"

def query_ollama(prompt):
    url = "http://localhost:11411/v1/query"
    headers = {'Content-Type': 'application/json'}
    data = {"model": "llama2", "input": prompt}
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # Raises an exception for 4xx/5xx errors
        return response.json()
    except requests.exceptions.HTTPError as errh:
        return f"HTTP Error: {errh}"
    except requests.exceptions.ConnectionError as errc:
        return f"Error Connecting: {errc}"
    except requests.exceptions.Timeout as errt:
        return f"Timeout Error: {errt}"
    except requests.exceptions.RequestException as err:
        return f"OOps: Something Else {err}"

# Main Dashboard
def main():
    st.title("AI Agent for Web Search")

    # Step 1: File Upload or Google Sheets Connection
    st.header("1. Upload File or Connect to Google Sheets")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    google_sheet_url = st.text_input("Or enter Google Sheets URL")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
    elif google_sheet_url:
        data = fetch_google_sheet(google_sheet_url)
    else:
        st.warning("Please upload a file or connect a Google Sheet.")
        return

    st.write("Uploaded Data Preview:")
    st.dataframe(data)

    # Step 2: Select Column and Query Input
    st.header("2. Define Your Search Query")
    column = st.selectbox("Select the main column for entities", data.columns)
    prompt_template = st.text_area(
        "Enter your prompt template",
        "Extract the email address of {entity} from the following web results:\n{results}"
    )

    # Step 3: Perform Web Search and Extract Information
    st.header("3. Search and Extract Information")
    if st.button("Start Search"):
        results = []
        for entity in data[column]:
            query = prompt_template.format(entity=entity)
            search_results = perform_web_search(query)
            extracted_info = extract_information_with_llm(search_results, prompt_template)
            results.append({"Entity": entity, "Extracted Info": extracted_info})

        # Display Results
        results_df = pd.DataFrame(results)
        st.write("Extraction Results:")
        st.dataframe(results_df)

        # Step 4: Download Results
        st.download_button(
            "Download CSV",
            data=results_df.to_csv(index=False).encode('utf-8'),
            file_name="extraction_results.csv",
            mime="text/csv"
        )

        # Step 5: Option to Write Back to Google Sheets
        if st.button("Write Results to Google Sheets"):
            credentials = service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"],
                scopes=["https://www.googleapis.com/auth/spreadsheets"],
            )
            sheet_id = google_sheet_url.split("/")[5]
            service = build("sheets", "v4", credentials=credentials)
            sheet = service.spreadsheets()

            # Prepare data to write back
            results_to_write = results_df.values.tolist()
            sheet.values().update(
                spreadsheetId=sheet_id,
                range="Sheet1!A1",
                valueInputOption="RAW",
                body={"values": results_to_write},
            ).execute()
            st.success("Results successfully written back to Google Sheets!")


if __name__ == "__main__":
    main()
