
import tmdbsimple as tmdb
import pickle
import pandas as pd
import numpy as np
import time
import os
import sys

class DataPipeline():

    def __init__(self, data_path, key):

        tmdb.API_KEY = key
        self.search = tmdb.Search()
        self.data_path = data_path

    def execute(self):
        netflix_df = self.parse_netflix(self.data_path)
        movie_df = self.generate_movie_df(netflix_df)
        tv_df = self.generate_tv_df(netflix_df)

    def parse_netflix(self, data_path, show=True):

        """
        Parse the netflix dataframe to and whehter a row is a TV shows or Movie
        :param netflix_df:
        :param show:
        :return:
        """
        netflix_df = pd.read_csv(data_path)
        netflix_df['Date'] = pd.to_datetime(netflix_df['Date'])
        netflix_df.rename({'Date':'Date Watched'}, index=1, inplace=True)

        # Split title into parts
        title_splits = netflix_df['Title'].str.split(':', 2, expand=True)
        title_splits.rename({0: 'Show Name', 1: 'Season', 2: 'Episode Name'}, axis=1, inplace=True)
        title_splits['Episode Name'] =  title_splits['Episode Name'].str.strip().tolist()
        title_splits['TV_Show_flag'] = title_splits['Episode Name'].apply(lambda row: 'Movie' if row == None else 'TV_Show')
        # Combine and output
        netflix_df_full = pd.concat([netflix_df, title_splits], axis=1)

        if show:
            print("Total number of TV Show + Movies: ", netflix_df_full.shape[0])
            print("TV Show vs Movie")
            print("Dataframe shape: ", netflix_df_full.shape)

        return(netflix_df_full)

    def get_movie_API_results(self, movie_title):
        # Select requested fields from response
        normal_movie_fields = ['budget', 'homepage', 'imdb_id', 'overview', 'popularity' \
            , 'release_date', 'revenue', 'runtime', 'vote_average', 'vote_count']

        # Find the Movie in TMDB
        search_results = self.search.movie(query=movie_title)
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

    def generate_movie_df(self, netflix_df, pkl_path='../Data/all_movies_results_df.pkl', show=True):
        print("Starting Movie data pull ")

        if os.path.isfile(pkl_path):
            print("Existing pickle exists")
            with open(pkl_path, 'rb') as hnd:
                self.all_movies_results_df = pickle.load(hnd)
        else:
            print("Querying TMDB to create pickle file")
            movies = netflix_df[netflix_df['TV_Show_flag'] == 'Movie']
            # Fresh Query
            all_movies_results = movies['Title'].apply(self.get_movie_API_results)
            self.all_movies_results_df = pd.DataFrame.from_dict(all_movies_results.to_list(), orient='columns')
            with open(pkl_path, 'wb') as hnd:
                pickle.dump(self.all_movies_results_df, hnd)

        if show:
            print("Number of movies: ", self.all_movies_results_df.shape[0])
            missing_movies = self.all_movies_results_df[self.all_movies_results_df['budget'].isna()]
            print("Number of missing movies: ", missing_movies.shape[0])
            print(missing_movies['title_query'])

        return(self.all_movies_results_df)


    def get_TV_show_details(self,tv_show_name):

        # Search TMDB by TV show name
        search_result = self.search.tv(query=tv_show_name)
        time.sleep(0.3)
        # Prep output dict
        output= {}
        output['query_term'] = tv_show_name

        schema = ['n_production_companies', 'primary_production_co', 'runtime', 'release_date'
            , 'n_network', 'primary_network', 'homepage', 'overview', 'popularity', 'vote_average', 'vote_count'
            , 'in_production', 'type', 'status', 'number_of_seasons', 'number_of_episodes']

        # Parse search response
        if len(search_result['results']) == 0:
            # No results
            for k in schema:
                output[k] = np.nan
            return(output)

        temp_id = search_result['results'][0]['id']
        raw_tv_response=tmdb.TV(temp_id).info()

        # Parse response
        output['n_production_companies'] = len(raw_tv_response['production_companies'])
        output['primary_production_co'] = '' if len(raw_tv_response['production_companies']) == 0 else \
        raw_tv_response['production_companies'][0]['name']
        output['runtime'] = 0 if len(raw_tv_response) == 0 else raw_tv_response['episode_run_time'][0]
        output['release_date'] = raw_tv_response['first_air_date']
        output['n_network'] = len(raw_tv_response['networks'])
        output['primary_network'] = '' if len(raw_tv_response['networks']) == 0 else \
        raw_tv_response['networks'][0]['name']

        # Fill dictionary with response items directly
        normal_tv_fields = ['homepage', 'overview', 'popularity' \
            , 'vote_average', 'vote_count', 'name', 'id'
            , 'in_production', 'type', 'status', 'number_of_seasons', 'number_of_episodes']
        for field in normal_tv_fields:
            output[field] = raw_tv_response[field]

        return (output)

    def generate_tv_df(self, netflix_df, pkl_path='../Data/all_tv_results_df.pkl', show=True):

        print("Starting TV data pull ")

        if os.path.isfile(pkl_path):
            print("Existing pickle exists")
            with open(pkl_path, 'rb') as hnd:
                self.all_tv_results_df = pickle.load(hnd)
        else:
            # Parse netflix dataframe and add seasons columns
            shows = netflix_df[netflix_df['TV_Show_flag'] == 'TV_Show']
            temp_season_numb = shows['Season'].str.split('Season ', n=2, expand=True)[1]
            # TODO: Fix this regex
            shows["Season"] = temp_season_numb.str.extract('^(\d{0,2}$)?')
            # Group by TV Show
            count_per_tv_title = shows.groupby('Show Name').size().sort_values(ascending=False)
            unique_shows = pd.Series(count_per_tv_title.index, name='Show Name')
            raw_TV_show_details = unique_shows.apply(self.get_TV_show_details)
            TV_show_details = pd.DataFrame(raw_TV_show_details.to_list())
            # Merge back to all Shows df
            self.all_tv_results_df = pd.merge(shows, TV_show_details, how='left', left_on='Show Name', right_on='query_term')
            # Pickle
            with open(pkl_path, 'wb') as hnd:
                pickle.dump(self.all_tv_results_df, hnd)

        if show:
            print("Number of total shows watched: ", self.all_tv_results_df.shape[0])
            missing_movies = self.all_tv_results_df[self.all_tv_results_df['id'].isna()]
            print("Number of unique shows watched: ", len(self.all_tv_results_df['Show Name'].unique()))
            print("Number of missing TV shows: ", missing_movies.shape[0])
            print(missing_movies['Show Name'])


        return(self.all_tv_results_df)



if __name__ =='__main__':
    data_path = '../Data/NetflixViewingHistory.csv'
    with open('../Data/api_key.pkl', 'rb') as hnd:
        key = pickle.load(hnd)['api_key']

    data_pipeline = DataPipeline(data_path, key)
    data_pipeline.execute()


