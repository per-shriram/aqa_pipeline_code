import os
from pathlib import Path
from typing import List, Dict
import logging
from openai import OpenAI
import numpy as np
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
import html2text
import re
from email import policy
from email.parser import BytesParser
import eml_parser
import json
from pathlib import Path
# from llama_index.vector_stores.faiss.base import FaissVectorStore
import faiss
import pickle
import logging
import numpy as np

import pinecone
from pinecone import Pinecone, ServerlessSpec
from flask import Flask, request, jsonify
from typing import List, Dict

class KnowledgeBaseAPI:
    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder
        self.app = Flask(__name__)
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.route('/search', methods=['POST'])
        def search():
            data = request.json
            query = data.get('query', '')
            top_k = data.get('top_k', 5)
            
            # Generate embedding for query
            query_embedding = self.embedder.generate_embeddings([query])[-1]
            
            # Search vector database
            results = self.vector_store.search(query_embedding, top_k)
            
            # Format results for GPT consumption
            formatted_results = self.format_results(results)
            
            return jsonify({
                'query': query,
                'results': formatted_results,
                'total_results': len(formatted_results)
            })
    
    def format_results(self, results) -> List[Dict]:
        """Format search results for GPT consumption"""
        formatted = []
        for result in results:
            if hasattr(result, 'metadata'):  # Pinecone format
                formatted.append({
                    'content': result.metadata['content'],
                    'sender': result.metadata['sender'],
                    'subject': result.metadata['subject'],
                    'date': result.metadata['date'],
                    'relevance_score': result.score
                })
            else:  # FAISS format
                doc = result['document']
                formatted.append({
                    'content': doc['content'],
                    'sender': doc['metadata']['sender'],
                    'subject': doc['metadata']['subject'],
                    'date': doc['metadata']['date'],
                    'relevance_score': result['score']
                })
        
        return formatted
    
    def run(self, host='0.0.0.0', port=5001):
        self.app.run(host=host, port=port)

class PineconeVectorStore:
    def __init__(self, api_key: str, environment: str = "us-east-1-aws"):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = "corporate-knowledge-base"
        self.dimension = 1536  # for text-embedding-3-small
        
    def create_index(self):
        """Create Pinecone index if it doesn't exist"""
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        
        self.index = self.pc.Index(self.index_name)
    
    def upsert_documents(self, documents: List[Dict], batch_size: int = 100):
        """Upload document embeddings to Pinecone"""
        vectors = []
        for doc in documents:
            vector = {
                'id': doc['metadata']['chunk_id'],
                'values': doc['embedding'],
                'metadata': {
                    'content': doc['content'],
                    'sender': doc['metadata']['sender'],
                    'subject': doc['metadata']['subject'],
                    'date': doc['metadata']['date'],
                    'chunk_index': doc['metadata']['chunk_index']
                }
            }
            vectors.append(vector)
            
            if len(vectors) >= batch_size:
                self.index.upsert(vectors)
                vectors = []
        
        # Upload remaining vectors
        if vectors:
            self.index.upsert(vectors)
    
    def search(self, query_embedding: List[float], top_k: int = 5):
        """Search for similar documents"""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results


class FAISSVectorStore:
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to FAISS index"""
        embeddings = np.array([doc['embedding'] for doc in documents]).astype('float32')
        self.index.add(embeddings)
        self.documents.extend(documents)
    
    def search(self, query_embedding: List[float], top_k: int = 5):
        """Search for similar documents"""
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[-1], indices[-1])):
            if idx < len(self.documents):
                result = {
                    'score': float(distance),
                    'document': self.documents[idx]
                }
                results.append(result)
        
        return results
    
    def save(self, filepath: str):
        """Save index and documents"""
        faiss.write_index(self.index, f"{filepath}.index")
        with open(f"{filepath}.docs", 'wb') as f:
            pickle.dump(self.documents, f)
    
    def load(self, filepath: str):
        """Load index and documents"""
        self.index = faiss.read_index(f"{filepath}.index")
        with open(f"{filepath}.docs", 'rb') as f:
            self.documents = pickle.load(f)
            
def extract_email_content(eml_file_path):
    """Extract structured content from EML file"""
    with open(eml_file_path, 'rb') as f:
        raw_email = f.read()
    
    ep = eml_parser.EmlParser()
    parsed_eml = ep.decode_email_bytes(raw_email)
    
    # Extract relevant fields
    email_data = {
        'sender': parsed_eml['header'].get('from', ''),
        'recipients': parsed_eml['header'].get('to', []),
        'subject': parsed_eml['header'].get('subject', ''),
        'date': parsed_eml['header'].get('date', ''),
        'message_id': parsed_eml['header'].get('message-id', ''),
        'body_text': '',
        'body_html': ''
    }
    
    # Extract body content
    for body in parsed_eml.get('body', []):
        if body.get('content_type') == 'text/plain':
            email_data['body_text'] = body.get('content', '')
        elif body.get('content_type') == 'text/html':
            email_data['body_html'] = body.get('content', '')
    
    return email_data

def preprocess_email_content(email_data):
    """Clean and preprocess email content"""
    h = html2text.HTML2Text()
    h.ignore_links = False  # Preserve links for context
    h.body_width = 0  # Don't wrap lines
    
    # Convert HTML to clean text
    if email_data['body_html']:
        clean_text = h.handle(email_data['body_html'])
    else:
        clean_text = email_data['body_text']
    
    # Remove email artifacts
    clean_text = re.sub(r'=\n', '', clean_text)  # Remove soft line breaks
    clean_text = re.sub(r'=3D', '=', clean_text)  # Decode quoted-printable
    clean_text = re.sub(r'=20', ' ', clean_text)  # Decode spaces
    
    # Standardize whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    logger = logging.getLogger(__name__)
    logger.info("Preprocessed email content for %s", email_data['message_id'])
    logger.debug("Cleaned text: %s", clean_text[:100])  # Log first 100 chars for debugging
    # Ensure content is not empty
    if not clean_text:
        logger.warning("Empty content after preprocessing for %s", email_data['message_id'])
        clean_text = "No content available"
    # Ensure content is not too long
    if len(clean_text) > 10000:  # Limit to 10,000 characters
        logger.warning("Content too long for %s, truncating to 10,000 characters", email_data['message_id'])
        clean_text = clean_text[:10000]

    # Create structured document
    document = {
        'content': clean_text,
        'metadata': {
            'sender': email_data['sender'],
            'subject': email_data['subject'],
            'date': email_data['date'],
            'message_id': email_data['message_id'],
            'recipients': email_data['recipients']
        }
    }
    
    return document

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into semantically coherent chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Chunking documents... with size %d and overlap %d resulting", chunk_size, chunk_overlap)

    chunked_docs = []
    for doc in documents:
        # Split the main content
        chunks = text_splitter.split_text(doc['content'])
        logger.info("Document %s split into %d chunks", doc['metadata']['message_id'], len(chunks))
        
        for i, chunk in enumerate(chunks):
            # Create chunk with preserved metadata
            chunk_doc = {
                'content': chunk,
                'metadata': {
                    **doc['metadata'],
                    'chunk_id': f"{doc['metadata']['message_id']}_{i}",
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
            }
            chunked_docs.append(chunk_doc)
    
    return chunked_docs

class EmbeddingGenerator:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.logger = logging.getLogger(__name__)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        self.logger.info(f"Found {len(texts)} EML files to process")
        """Generate embeddings for a batch of texts"""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [data.embedding for data in response.data]
    
    def embed_documents(self, documents: List[Dict]) -> List[Dict]:
        """Add embeddings to document chunks"""
        texts = [doc['content'] for doc in documents]
        embeddings = self.generate_embeddings(texts)
        
        for doc, embedding in zip(documents, embeddings):
            doc['embedding'] = embedding
            doc['embedding_model'] = self.model
        
        return documents

class EMLKnowledgeBasePipeline:
    def __init__(self, openai_api_key: str, pinecone_api_key: str = None):
        self.embedder = EmbeddingGenerator(openai_api_key)
        
        if pinecone_api_key:
            self.vector_store = PineconeVectorStore(pinecone_api_key)
            self.vector_store.create_index()
        else:
            self.vector_store = FAISSVectorStore()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def process_eml_directory(self, eml_directory: str) -> List[Dict]:
        """Process all EML files in a directory"""
        eml_files = list(Path(eml_directory).glob("*.eml"))
        self.logger.info(f"Found {len(eml_files)} EML files to process")
        
        all_documents = []
        for eml_file in eml_files:
            try:
                # Extract email content
                email_data = extract_email_content(eml_file)
                self.logger.info(f"Processing {eml_file.name} from {email_data['date']} by {email_data['sender']}")
                
                # Preprocess content
                document = preprocess_email_content(email_data)
                self.logger.info(f"Preprocessed document for {email_data['message_id']}")
                all_documents.append(document)
                
            except Exception as e:
                self.logger.error(f"Error processing {eml_file}: {e}")
        
        self.logger.info(f"Successfully processed {len(all_documents)} emails")
        return all_documents
    
    def build_knowledge_base(self, eml_directory: str, chunk_size: int = 1000):
        """Build complete knowledge base from EML files"""
        # Process EML files
        documents = self.process_eml_directory(eml_directory)
        
        # Chunk documents
        self.logger.info("Chunking documents...")
        chunked_docs = chunk_documents(documents, chunk_size=chunk_size)
        self.logger.info(f"Created {len(chunked_docs)} chunks")
        
        # Generate embeddings
        self.logger.info("Generating embeddings...")
        embedded_docs = self.embedder.embed_documents(chunked_docs)
        
        # Store in vector database
        self.logger.info("Storing in vector database...")
        if isinstance(self.vector_store, PineconeVectorStore):
            self.vector_store.upsert_documents(embedded_docs)
        else:
            self.vector_store.add_documents(embedded_docs)
            self.vector_store.save("knowledge_base")
        
        self.logger.info("Knowledge base built successfully!")
        return embedded_docs

# Usage example
pipeline = EMLKnowledgeBasePipeline(
    openai_api_key=os.environ.get("OPENAI_API_KEY") 
    # , None # pinecone_api_key="your-pinecone-key"  # Optional
    )

# Build knowledge base
documents = pipeline.build_knowledge_base("..")

# Create API server
api = KnowledgeBaseAPI(pipeline.vector_store, pipeline.embedder)
api.run()
