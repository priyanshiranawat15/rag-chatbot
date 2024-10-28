# RAG

This is a simple implementation of Retreival Augmented Generation. Basic usage includes loading the files into the database and then retreiving the files using the RAG model.

## Requirements

- Python 3.10
- Postgres 15
- pgvector extension for Postgres ([here](github.com/pgvector/pgvector))

## Usage

1. Firstly install the requirements using the following command:

```bash
pip install -r requirements.txt
```

Now there are two options:

- Load files into the database using the following command:

```bash
 python file_laoder.py
```

- Run the RAG model using the following command:

```bash
 python file_retreiver.py
```

## file_loader.py

This is a script for loading the files into the database. Pass the folder name as asked relative to this file. Once the folder is passed, you'll have option for `update` to update in preexisting collection,`delete` to delete a collection and `create` to create a collection in the database.

## file_retreiver.py

This is a script for retreiving the files from the database. Pass the query as asked and you'll get the relevant files. Flow:

1. Pass the query
2. The query is vectorized by embedding model.
3. The vector is passed to the database and the relevant files are retreived.
4. The documents are then passed to the chat model and instructions are given to the model for context and question answering.
