
import tmdbsimple as tmdb
import pickle
import pandas as pd
import numpy as np
import time
import os
import sys

with open('../Data/api_key.pkl', 'rb') as hnd:
    tmdb.API_KEY = pickle.load(hnd)['api_key']

search= tmdb.Search()

def parse_netflix(netflix_df, show=True):

    """
    Parse the netflix dataframe to and whehter a row is a TV shows or Movie
    :param netflix_df:
    :param show:
    :return:
    """
    netflix_df['Date'] = pd.to_datetime(netflix_df['Date'])
    netflix_df.rename({'Date':'Date Watched'}, index=1, inplace=True)

    # Split title into parts
    title_splits = netflix_df['Title'].str.split(':', 2, expand=True)
    title_splits.rename({0: 'Show Name', 1: 'Season', 2: 'Episode Name'}, axis=1, inplace=True)
    title_splits['TV_Show_flag'] = title_splits['Episode Name'].apply(lambda row: 'Movie' if row == None else 'TV_Show')
    # Combine and output
    netflix_df_full = pd.concat([netflix_df, title_splits], axis=1)

    if show:
        print("Total number of TV Show + Movies: ", netflix_df_full.shape[0])
        print("TV Show vs Movie")
        print(netflix_df_full['TV_Show_flag'].value_counts())
        print("Dataframe shape: ", netflix_df_full.shape)

    return(netflix_df_full)


def get_movie_API_results(movie_title):
    # Select requested fields from response
    normal_movie_fields = ['budget', 'homepage', 'imdb_id', 'overview', 'popularity' \
        , 'release_date', 'revenue', 'runtime', 'vote_average', 'vote_count']

    # Find the Movie in TMDB
    search_results = search.movie(query=movie_title)
    n_results = len(search_results['results'])
    #     print("N Results: ", n_results)
    if n_results == 0:
        movie_results = {key: np.nan for key in normal_movie_fields}
        movie_results['Number of Search Results'] = n_results
        movie_results['title_query'] = movie_title
        return (movie_results)

    temp_id = search_results['results'][0]['id']
    full_movie_results = tmdb.Movies(temp_id)

    assert (set(normal_movie_fields).difference(set(full_movie_results.info().keys())) == set()) \
        , 'Movie result schema is missing a field'
    movie_results = {attr: getattr(full_movie_results, attr) for attr in normal_movie_fields}
    # TODO Fix genre parsing

    # Append number of search results (incase there are multiple and we choose the wrong one)
    movie_results['Number of Search Results'] = n_results
    movie_results['title_query'] = movie_title

    time.sleep(0.6)
    return (movie_results)

def generate_movie_df(netflix_df, pkl_path='../Data/all_movies_results_df.pkl', show=True):

    if os.path.isfile(pkl_path):
        print("Existing pickle exists")
        with open(pkl_path, 'rb') as hnd:
            all_movies_results_df = pickle.load(hnd)
    else:
        print("Querying TMDB to create pickle file")
        movies = netflix_df[netflix_df['TV_Show_flag'] == 'Movie']
        # Fresh Query
        all_movies_results = movies['Title'].apply(get_movie_API_results)
        all_movies_results_df = pd.DataFrame.from_dict(all_movies_results.to_list(), orient='columns')
        with open(pkl_path, 'wb') as hnd:
            pickle.dump(all_movies_results_df, hnd)

    if show:
        print("Number of movies: ", all_movies_results_df.shape[0])
        missing_movies = all_movies_results_df[all_movies_results_df['budget'].isna()]
        print("Number of missing movies: ", missing_movies.shape[0])
        print(missing_movies['title_query'])

    return(all_movies_results_df)


