#!/usr/bin/env python3
import pandas as pd
import sqlite3
import sys
import random
import collections

def getDatabase(conn, content=True, title=False, tableName="clean"):
    if content:
        fieldName = "content"
    else:
        fieldName = "title"
    news = pd.read_sql(f"SELECT {fieldName} FROM {tableName} WHERE LENGTH({fieldName}) >= 6000;", conn)
    return news, fieldName
    
def getConnection(PATH=None):
    if True:
        """
        Creates a database connection to the 099.db database file
        located at ~/./all-the-news.db
        """
        PATH="news.db" if PATH == None else PATH
        try:
            connection = sqlite3.connect(PATH)
            assert(connection is not None)
            return connection
        except ConnectionRefusedError:
            print(ConnectionRefusedError)
        return None

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
    local_connection = getConnection(PATH="news.db") if (path == None) else getConnection(PATH=path)
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
"""
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
"""

def main(path=None):
    path = "news.db" if (path == None) else path
    yield collections.deque(word for word in next(extract()).split(' ') if word !='')


def newArticle(allNews):
    def wrapper():
        yield collections.deque(word for word in next(allNews).split(' ') if word !='')
    def getNext(generator_obj):
        try:
            return next(generator_obj)
        except StopIteration:
            return getNext(wrapper())
        except TypeError:
            return getNext(wrapper())
        except Exception:
            raise
    return getNext(wrapper())

def nextWords(article, allNews, nextChunk=None, wordcount = 50, wc=-1, dropFirst=False):
    """Example usage:
    allNews = dbExtract.extract()
    new_article = dbExtract.newArticle(allNews)

    Steps:
    1. article.popleft() first {wordcount} words from article
    2. 
    """
    def fillup(nextChunk=None, wc=wc):
        nextChunk = collections.deque(maxlen=wordcount) if nextChunk == None else nextChunk
        wc = 0 if len(nextChunk) == 0 else len(nextChunk)-1
        while wc < wordcount:
            try:
                nextChunk.append(article.popleft())
                wc += 1
                if wc == wordcount:
                    return nextChunk
                else:
                    return fillup(nextChunk=nextChunk, wc=wc)
            except IndexError:
                wc += 1
                return nextWords(newArticle(allNews), 
                        allNews=allNews, 
                        nextChunk=nextChunk, 
                        wordcount=wordcount,
                        wc=wc)
    if dropFirst:
        nextChunk.popleft()
    return fillup(nextChunk, wc)

def saveGenerator(_allNews):
    with open("allNews.txt", "w") as f:
        f.writelines(_allNews)

def openGenerator():
    with open("allNews.txt", "r") as f:
        return (_ for _ in f.readlines())
