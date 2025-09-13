from flask import Flask, request, render_template, jsonify 
from models.vector_store import VectorStore 
from services.storage_service import S3Storage 
from services.llm_service import LLMService 
from config import Config 
import os
from langshain.document_loaders import TextLoader, PyPDFLoader 
from langshain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import logging


app = Flask(__name__)
vector_store = VectorStore(Config-VECTOR_OB_PATH)
storage_service = S3Storage()
llm_service = LLMService(vector_store)

@app. route("/")
def index():
    return render_template( 'index.html')

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
