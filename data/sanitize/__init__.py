#!/usr/bin/env python3
import pandas as pd
import re
import sqlite3

def preprocess(articleString):
    try:
        articleString = articleString.lower()
        articleString = re.sub(r"([.,!?])", r" \1 ", articleString)
        articleString = re.sub(r"[^a-zA-A.,!?]+", r" ", articleString)
        return articleString
    except Exception:
        pass

def collect_sanitized(aG):
    collection = []
    for article in aG:
        collection.append(preprocess(article))
    return pd.concat([pd.DataFrame([c], columns=['content']) for c in collection],ignore_index=True)

def write_local(dataframe, PATH):
    Local_Connection = sqlite3.connect(PATH)
    assert(Local_Connection is not None)
    dataframe.to_sql(name='clean', con=Local_Connection)

def data_Wrangle(Path, articleGenerator):
    write_local(collect_sanitized(aG=articleGenerator), PATH=Path)
