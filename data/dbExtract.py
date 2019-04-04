#!/usr/bin/env python3
import pandas as pd
import sqlite3
import sys
import random

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

def idCheck(idInt):
    PATH = "used.txt"
    with open(PATH, "r+") as f:
        try:
            used = f.read()
            try:
                assert(idInt not in used)
                return idInt
            except AssertionError:
                f.write(idInt)     
                return -1
        except Exception:
            pass

def writeUsed(path, *used):
    PATH = "used.txt"
    with open(path, "r+") as f:
        for u in used:
            f.write(u)

def partition(path=None, default_size=179599):
    path = path if (path != None) else "news.db"
    availabe_IDs = [id if not idCheck(id)==-1 for id in range(0,default_size)]
    random.shuffle(availabe_IDs)
    with open(path, "r+") as f:
        try:
            used = f.read()
        except Exception:
            used = []
    rand = random.sample(availabe_IDs, round(0.1*default_size / 3, ndigits=0))
    # pass on rand to be the ID's that get matched with a SQL query
    # write these to their own unique temporary db.file

