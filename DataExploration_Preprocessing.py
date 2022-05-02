#Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from nltk.corpus import stopwords
stop = stopwords.words('english')

#Load the data
current_path = os.path.dirname(os.path.realpath(__file__))
data_path = "Data\imdb_top_1000.csv"
df_imdb = pd.read_csv(f"{current_path}\\{data_path}")

#get an overview of the data
# print("First Five Rows of DataFrame:")
# print(df_imdb.head())
# print("Last Five Rows of DataFrame:")
# print(df_imdb.tail())
# print("Random 10 row sample of Dataframe:")
# print(df_imdb.sample(10))
# print("Length of Dataframe:")
#print(len(df_imdb))

#identify variable type
# print("Column Types of Dataframe:")
# print(df_imdb.dtypes)
# print("Information of DataFrame:")
# print(df_imdb.info())
# print("Descrptive statistics of Dataframe:")
# print(df_imdb.describe())

#Understand various summary statistics of the data
include =['object', 'float', 'int']
df_imdb.describe(include=include)
#print(df_imdb.describe())

# Finding Missing Values
#print(df_imdb.isnull().sum())

# Preprocessing
def columns_to_keep(df,listCols):
    cols_to_keep = listCols
    df = df[cols_to_keep]
    return df

def text_cleaning(df,listCols):

    for i in listCols:
        try:
            df[i] = df[i].str.lower()
            df[i] = df[i].str.strip()
            df[i] = [re.sub(r"[^a-zA-Z0-9 ]", "", str(x)) for x in df[i]]
        except Exception as e:
            print(e)

def convert_ratings_into_tiers(x):
    if x < 8:
        return 'tier 3'
    elif x >= 8 and x < 9:
        return 'tier 2'
    else:
        return 'tier 1'

# merge dataframe columns together
def merge_description(df, col_list):
    return df[col_list].apply(lambda x: ' '.join(x), axis=1)
