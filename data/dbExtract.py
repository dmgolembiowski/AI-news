#!/usr/bin/env python3
import pandas as pd
import sqlite3
import sys

def getDatabase(conn, content=True, title=False, tableName="longform"):
    if content:
        fieldName = "content"
    else:
        fieldName = "title"
    news = pd.read_sql(f"SELECT {fieldName} FROM {tableName};", conn)
    return news, fieldName
    
def getConnection(local=True, PATH="all-the-news.db"):
    if local:
        """
        Creates a database connection to the 099.db database file
        located at ~/./all-the-news.db
        """
        PATH="all-the-news.db"
        try:
            connection = sqlite3.connect(PATH)
            assert(connection is not None)
            return connection
        except ConnectionRefusedError:
            print(ConnectionRefusedError)
        return None
    else:
        print('Undefined value for "local" argument. Exiting program...')
        raise AttributeError
        sys.exit()

def extract(path=None):
    """
    Usage:
            news = dbVectorize.main()
            Article1 = next(news)
            import spacy
            # Download package:
            # python -m spacy download en_core_web_sm
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(Article1)
            for chunk in doc.noun_chunks:
                print(f'{chunk} - {chunk.label_}')
    """
    local_connection = getConnection() if (path == None) else getConnection(PATH=path)
    news, fieldName = getDatabase(conn=local_connection)
    return (entry for entry in news[fieldName])

