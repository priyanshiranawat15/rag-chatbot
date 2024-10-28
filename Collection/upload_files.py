import os
import pypdf
import warnings
warnings.filterwarnings("ignore")
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
import psycopg2
from dotenv import load_dotenv
from langchain_community.vectorstores import PGVector
import os
load_dotenv()
import numpy as np


def load_from_folder(folder_path):
    doc_list = []
    for file in os.listdir(folder_path):
        file_list = load_file(folder_path=folder_path,file=file)
        for doc in file_list:
            doc_list.append(doc)
    print("Total number of pages loaded:" , len(doc_list))
    return doc_list

def load_file(folder_path,file):
    doc_list = []
    if file.endswith(".pdf"):
        loader = PyPDFLoader(f"{folder_path}/{file}")
        x = loader.load_and_split()
        doc_list= x
    return doc_list

def embed_and_store_collection(doc_list, model, collection_name, connection_string, exists=False):
    """Stores a list of documents into a pgvector database.

    Args:
        doc_list: A list of documents (strings) to store.
        embedding_model: A HuggingFaceEmbeddings model for generating embeddings.
        collection_name: The name of the collection to store the documents in.
        connection_string: The connection string to the PostgreSQL database.
    """

    db = PGVector.from_documents(
        embedding=model,
        documents=doc_list,
        collection_name=collection_name,
        connection_string=connection_string,
    )
    print(f"New database and collection '{collection_name}' created")
    return db

def embed_and_store(doc_list, model, collection_name, connection_string):
    """Stores a list of documents into a pgvector database.

    Args:
        doc_list: A list of documents (strings) to store.
        embedding_model: A HuggingFaceEmbeddings model for generating embeddings.
        collection_name: The name of the collection to store the documents in.
        connection_string: The connection string to the PostgreSQL database.
    """

    db = PGVector(
        embedding_function=model,
        collection_name=collection_name,
        connection_string=connection_string
    )
    print(f"{db} collection connected")
    try:
        db.add_documents(doc_list)
    except Exception as e:
        print(f"Error storing documents in the database: {e}")
    return db

def _embed_and_store(doc_list, embedding):
    try:
        print("Connecting to the database")

        conn = psycopg2.connect(database=os.getenv('DB_NAME'),
                                user=os.getenv('DB_USER'),
                                password=os.getenv('DB_PASSWORD'),
                                host=os.getenv('DB_HOST'),
                                port= os.getenv('DB_PORT')
        )
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None
    cursor = conn.cursor()

    print("connection established")
    for doc in doc_list:
        vector = embedding.embed_query(doc.page_content)
        id = f"{doc.metadata['source'].split('/')[-1]}_{doc.metadata['page']}"
        print(f"Document {id} embedded")

        print(f"Document {id} indexed")
        print(f"vector shape : {len(vector)} vector type : {type(vector)}")

        print(f"{doc.page_content[:1]}...{doc.page_content[-10:]}")
        # Store the document in the database
        #remove the single quotes from the content
        content = doc.page_content.replace("'", "")
        try:
            cursor.execute(f"""
                    INSERT INTO documents (content, embedding)
                    VALUES ('{content}', ARRAY{vector}) 
                """) 
        except Exception as e:
            print(f"Error storing document {id} in the database: {e}")
        print(f"Document {id} stored in the database")
    conn.commit()
    cursor.close()
    print("connection closed")

def get_connection_string():
    connstring = PGVector.connection_string_from_db_params(
        driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
        host=os.environ.get("PGVECTOR_HOST", "localhost"),
        port=int(os.environ.get("PGVECTOR_PORT", "5433")),
        database=os.environ.get("PGVECTOR_DATABASE", "postgres"),
        user=os.environ.get("PGVECTOR_USER", "postgres"),
        password=os.environ.get("PGVECTOR_PASSWORD", "admin"),
    )
    return connstring

def delete_collection(collection_name, connection_string):
    db = PGVector(
        embedding_function=None,
        collection_name=collection_name,
        connection_string=connection_string
    )
    db.delete_collection()
    print(f"Collection '{collection_name}' deleted")

def embedding_model():
    model = HuggingFaceEmbeddings()
    return model