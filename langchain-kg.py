from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.chat_models.ollama import ChatOllama

from langchain_core.language_models import BaseLanguageModel
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from text_to_graph.text_to_graph import LLMDoc2GraphTransformer

from gen_ai_hub.proxy.core import set_proxy_version
from gen_ai_hub.proxy import GenAIHubProxyClient
from gen_ai_hub.proxy.langchain import init_llm

import logging, os, pickle
from typing import Union

MODEL_NAME = "phi3:14b-medium-128k-instruct-q6_K"
# MODEL_NAME = "phi3:14b-medium-4k-instruct-q8_0"
# MODEL_NAME = "llama3:8b"
# MODEL_NAME = "command-r:35b-v0.1-q3_K_S"
# MODEL_NAME = "llama3-gradient:8b"
# MODEL_NAME = "wizardlm2:7b"
# MODEL_NAME = "gemma:7b-instruct"
# MODEL_NAME = "phi3:3.8b"


import os, json
from datetime import datetime
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import requests

# -------------- Env. Variables --------------->>>
CLIENT_ID = os.environ.get("AICORE_OLLAMA_CLIENT_ID")
CLIENT_SECRET = os.environ.get("AICORE_OLLAMA_CLIENT_SECRET")
TOKEN_URL = os.environ.get("AICORE_OLLAMA_AUTH_URL")
API_URL = os.environ.get("AICORE_OLLAMA_API_BASE")
RESOURCE_GROUP = os.environ.get("AICORE_OLLAMA_RESOURCE_GROUP")
DEPLOYMENT_NAME = os.environ.get("DEPLOYMENT_NAME")
RETRY_TIME = int(os.environ.get("RETRY_TIME","30"))
DEPLOYMENT_API_PATH = "/lm/deployments" 

# OAuth2 token
TOKEN = {
            "ollama-2-server": {
                "envParams": {
                    "token": "AICORE_TOKENURL",
                    "id": "OPENAI_CLIENTID",
                    "sec": "OPENAI_CLIENTSECRET"
                },
                "token": {}
                }
        }

def get_token(service: str) -> str:
    get_token = False
    client = BackendApplicationClient(client_id=CLIENT_ID)
    # create an OAuth2 session
    oauth = OAuth2Session(client=client)
    if TOKEN[service]['token'] == {}:
        get_token = True
    elif datetime.now() > datetime.fromtimestamp(TOKEN[service]['token']['expires_in']):
        get_token = True
    if get_token:
        TOKEN[service]['token'] = oauth.fetch_token(token_url=TOKEN_URL, client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
        TOKEN[service]['token']['expires_at'] = datetime.now().timestamp()+TOKEN[service]['token']['expires_in']
    return f"Bearer {TOKEN[service]['token']['access_token']}"

def get_baseurl()->str:
    """ Retrieves the AI Core deployment URL """
    # Request an access token using client credentials
    access_token = get_token(DEPLOYMENT_NAME)
    
    headers = {
        'Authorization': access_token,
        'AI-Resource-Group': RESOURCE_GROUP
    }
    res = requests.get(API_URL+DEPLOYMENT_API_PATH, headers=headers)
    j_data = res.json()
    for resource in j_data["resources"]:
        if resource["scenarioId"] == DEPLOYMENT_NAME:
            if resource["deploymentUrl"] == "":
                print(f"Scenario '{DEPLOYMENT_NAME}' was found but deployment URL was empty. Current status is '{resource['status']}', target status is '{resource['targetStatus']}'. Retry in {str(RETRY_TIME)} seconds.")
            else:
                print(f"Scenario '{DEPLOYMENT_NAME}': Plan '{resource['details']['resources']['backend_details']['predictor']['resource_plan']}', modfied at {resource['modifiedAt']}.")
            return f"{resource['deploymentUrl']}/v1"
   
def get_models(self)->list:
    """ Retrieves list of available models in Ollama instance """
    model_url = self.base_url + "/api/tags"
    access_token = get_token(DEPLOYMENT_NAME)
    
    headers = {
        'Authorization': access_token,
        'AI-Resource-Group': RESOURCE_GROUP
    }
    res = requests.get(model_url, headers=headers)
    j_data = res.json()
    return j_data

def pull_model(self, model: str)->list:
    """ Pulls a model through Ollama """
    pull_url = self.base_url + "/api/pull"
    access_token = get_token(DEPLOYMENT_NAME)
    
    headers = {
        'Authorization': access_token,
        'AI-Resource-Group': RESOURCE_GROUP
    }
    result_list = []  # To store the parsed JSON objects
    with requests.post(pull_url, headers=headers, stream=True, json={"name": model}) as res:
        if res.status_code == 200:
            for line in res.iter_lines(chunk_size=1024, decode_unicode=True):
                if line:
                    # Parse the JSON object from the NDJSON line
                    json_object = json.loads(line)                        
                    # Process or store the parsed JSON object as needed
                    print(f"Model: {model}: {str(json_object)}")                        
                    # Append the parsed JSON object to the result list
                    result_list.append(json_object)
            return result_list
        else:
            # Handle error cases
            print(f"Error: {res.status_code}, {res.text}")    
            return []


def get_llm(type: str, model: str, temperature: float = 0.0, top_p: float = 0.9)->Union[BaseLanguageModel, ChatOllama]:
    if type == "ollama":
        access_token = get_token(DEPLOYMENT_NAME)
        headers = {
            'Authorization': access_token,
            'AI-Resource-Group': RESOURCE_GROUP
        }
        chat_ollama_kwargs = {
            "base_url": get_baseurl(),
            "headers": headers,
            "model": model,
            "temperature": temperature,
            "top_p": top_p
        }
        if model.startswith("llama3-gradient"):
            chat_ollama_kwargs["options"] = {"num_ctx": 64000 } # Context window size
        return ChatOllama(**chat_ollama_kwargs)
    elif type == "genaihub":        
        set_proxy_version('gen-ai-hub') # for an AI Core proxy
        genai_hub_client = GenAIHubProxyClient()
        init_llm_kwargs = {
            "model_name": MODEL_NAME,
            "proxy_client": genai_hub_client,
            "stream": True # Testing if that works
        }
        return init_llm(**init_llm_kwargs)
    else:
        return None   


def convert_to_text(file: list, filename: str, chunk_size: int, chunk_overlap: int, use_ocr: bool)->list[Document]:
    """ Converts file to text """
    if file.endswith('.pdf'):
        loader = PyPDFLoader(file_path=file, extract_images=use_ocr)
    elif file.endswith('.txt'):
        loader = TextLoader(file_path=file)
    else:
        raise ValueError('File format not supported. Please provide a .pdf or .docx file.')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                chunk_overlap=chunk_overlap, 
                                                length_function=len, 
                                                is_separator_regex=False
                                                )
    documents = loader.load_and_split(text_splitter)
    for doc in documents:
        if doc.metadata.get("source", None) != None:
            doc.metadata["source"]=filename
        if doc.metadata.get("page", None) != None: # increment by 1 to avoid page 0
            doc.metadata["page"] += 1
    return documents
        

def main()->None:
    log_level = int(os.environ.get("APPLOGLEVEL", logging.ERROR))
    doc_folder = os.environ.get("DOCS_FOR_ANALYSIS_FOLDER", "./docs-for-analysis")
    pickle_folder = os.environ.get("PICKLEFILE_FOLDER", "./docs-for-analysis")
    chunksize_nodes = int(os.environ.get("CHUNKSIZE_NODES", 500))
    chunksize_edges = int(os.environ.get("CHUNKSIZE_EDGES", 1000))
    chunkoverlap_nodes = int(os.environ.get("CHUNKSIZE_NODES", 50))
    chunkoverlap_edges = int(os.environ.get("CHUNKSIZE_NODES", 100))
    if log_level < 10: log_level = 40
    logging.basicConfig(level=log_level)
    logging.getLogger("requests").setLevel(logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.info("Connecting to SAP AI Core hosted LLMs...")

# Top p with 5 docs
# 0.0 good! 25 entries
# 0.1 good! 30 entries
# 0.2 ok. 23 entries
# 0.3 good results. 29 entries
# 0.4 good results. 29 entries
# 0.5 same? 29 entries.
# 0.7 good. 25 entries.
# 0.9 good. 23 entries.

    llm = get_llm(type="ollama", model=MODEL_NAME, top_p=0.1, temperature=0.0)
    logger.info(f"Using model {MODEL_NAME}.")
    
    # Load PDF document
    filename = "SAP Service General Terms and Conditions.pdf"
    documents_for_nodes = convert_to_text(file=f"{doc_folder}/{filename}", filename=filename, chunk_size=chunksize_nodes, chunk_overlap=chunkoverlap_nodes,use_ocr=False)
    documents_for_edges = convert_to_text(file=f"{doc_folder}/{filename}", filename=filename, chunk_size=chunksize_edges, chunk_overlap=chunkoverlap_edges,use_ocr=False)
    
    documents_for_nodes = documents_for_nodes[:5]
    documents_for_edges = documents_for_edges[:2]
    logger.info(f"Split document {filename} into {len(documents_for_nodes)} documents for nodes and {len(documents_for_edges)} for edges.")
    
    llm_transformer = LLMDoc2GraphTransformer(llm=llm, pickle_folder=pickle_folder, store_to_disk=True)
    graph_documents = llm_transformer.convert_to_graph_documents(docs_nodes=documents_for_nodes, docs_edges=documents_for_edges)
    # Save the graph_documents object to disk
    logger.info(f"{len(graph_documents)} documents were extracted. Storing object to disk.")
    with open(f"{pickle_folder}/{filename}_graph.pkl", "wb") as f:
        pickle.dump(graph_documents, f)
    logger.info("Extraction finished.")
    
if __name__ == "__main__":
    main()