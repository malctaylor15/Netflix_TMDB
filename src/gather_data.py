
import tmdbsimple as tmdb
import pickle
import pandas as pd
import os
import sys

with open('../Data/api_key.pkl', 'rb') as hnd:
    tmdb.API_KEY = pickle.load(hnd)['api_key']

def parse_netflix(netflix_df, show=True):
    netflix_df['TV_Show_flag'] = netflix_df.Title.apply(lambda x: 'TV Show' if ': Season' in x else 'Movie')
    netflix_df['Date'] = pd.to_datetime(netflix_df['Date'])
    netflix_df.rename({'Date':'Date Watched'}, index=1, inplace=True)

    # Split title into parts
    title_splits = netflix_df['Title'].str.split(':', 2, expand=True)
    title_splits.rename({0: 'Show Name', 1: 'Season', 2: 'Episode Name'}, axis=1, inplace=True)
    # Combine and output
    netflix_df_full = pd.concat([netflix_df, title_splits], axis=1)

    if show:
        print("Total number of TV Show + Movies: ", netflix_df.shape[0])
        print("TV Show vs Movie")
        print(netflix_df['TV_Show_flag'].value_counts())
        print("Dataframe shape: ", netflix_df_full.shape)

    return(netflix_df_full)

