import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import tensorflow as tf
import tensorflow_hub as hub
from DataExploration_Preprocessing import columns_to_keep,text_cleaning,\
    convert_ratings_into_tiers,merge_description,stop

import os
current_path = os.path.dirname(os.path.realpath(__file__))
data_path = "Data\imdb_top_1000.csv"

modelpath_list = [f"{current_path}\\Models\\all-distilroberta-v1",f"{current_path}\\Models\\all-mpnet-base-v2",f"{current_path}\\Models\\all-MiniLM-L6-v2",
                  f"{current_path}\\Models\\multi-qa-distilbert-cos-v1"]


def generateEmbeddings(modelpath,sentenceList):
    model = SentenceTransformer(modelpath)
    embeddings = model.encode(sentenceList)

    return embeddings


def recommend(movie_id, n_rec, features, df):

    """
    :return: recommendation list
    """

    # Get the index of the product
    idx = df[df['id'] == movie_id].index[0]

    # Get the pair-wsie similarity scores of all products with that product
    cs = cosine_similarity(features[idx].reshape(1, -1), features)
    rec_index = np.argsort(cs)[0][::-1]

    # Generate top-N recommendations
    rec = list()

    i = 0
    while i <= n_rec:
        rec_ith = df.iloc[rec_index[i]]['id']
        rec.append(
                df.loc[df['id'] == rec_ith, ['id', 'series_title', 'genre', 'tier']].values[0])
        i += 1

    rec_df = pd.DataFrame(rec, columns=['id', 'series_title', 'genre', 'tier'])

    return rec_df


if __name__ == '__main__':
    df_imdb = pd.read_csv(f"{current_path}\\{data_path}")

    col_keep = ['id', 'series_title', 'genre', 'imdb_rating', 'overview', 'director']
    df_imdb = columns_to_keep(df_imdb,col_keep)
    df_imdb.dropna(inplace=True)

    text_cleaning(df_imdb,['series_title', 'genre', 'overview', 'director'])
    df_imdb['tier'] = df_imdb['imdb_rating'].apply(lambda x: convert_ratings_into_tiers(x))

    cols_to_merge = ['series_title', 'genre', 'overview', 'director', 'tier']

    df_imdb = df_imdb.assign(description=lambda x: merge_description(x, cols_to_merge))

    df_imdb['description'] = df_imdb['description'].apply(lambda x: ' '.join([w for w in x.split() if w not in (stop)]))

    # Model Recommendations

    movie_id = 5
    rec_number = 10

    for model in modelpath_list:
        sentence_embeddings = generateEmbeddings(model,df_imdb['description'].values)
        sentence_embeddings = normalize(sentence_embeddings, axis=1, norm='l2')

        model_name = model.split("\\")[-1]
        print(f"Model Name: {model_name}")

        rec_df = recommend(
            movie_id=5,
            n_rec=5,
            features=sentence_embeddings,
            df=df_imdb
        )

        rec_df.to_csv(f"{current_path}\\Recommendations\\{model_name}_{movie_id}_{rec_number}.csv",index=False)