from Collection.upload_files import delete_collection,load_from_folder, embed_and_store_collection, embed_and_store, get_connection_string
from langchain.embeddings import HuggingFaceEmbeddings
import os

root = os.path.dirname(__file__)

model = HuggingFaceEmbeddings()
connection_string = get_connection_string()

response = input('Enter the action to perform: ')

if response.lower() == 'update':
    folder_name = input('Enter the folder path: ')
    folder_path = os.path.join(root,folder_name)
    collection_name = input('Enter the name of the collection: ')
    print("Collection name : ",collection_name)
    doc_list = load_from_folder(folder_path)
    db = embed_and_store(
        model=model,
        doc_list=doc_list,
        collection_name=collection_name,
        connection_string=connection_string,
    )
    print("Collection updated")
    exit()

if response.lower() == 'create':
    folder_name = input('Enter the folder path: ')
    folder_path = os.path.join(root,folder_name)
    collection_name = input('Enter the name of the collection: ')
    print("Collection name : ",collection_name)
    doc_list = load_from_folder(folder_path)
    db = embed_and_store_collection(
        model=model,
        doc_list=doc_list,
        collection_name=collection_name,
        connection_string=connection_string,
    )
    print("Collection created")
    exit()

if response.lower() == 'delete':
    response = input('enter the name of the collection to delete: ')
    delete_collection(response,connection_string)
    exit()

