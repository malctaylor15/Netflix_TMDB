
import tmdbsimple as tmdb
import pickle
import pandas as pd
import numpy as np
import re
import uuid
import datetime
import sqlite3
from fuzzywuzzy import fuzz
# from tqdm import tqdm
from tqdm.autonotebook import tqdm

import pdb
import sys
tqdm.pandas()

def get_db_tables(con):
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tbls = cursor.fetchall()
    tbls = [x[0] for x in tbls]
    return(tbls)

def get_series_watched_gb(df):

    out = {}
    out['Number of Episodes'] = df.shape[0]
    out['First Episode Watcheed'] = df['Date Watched'].min()
    out['Last Episode Watched'] = df['Date Watched'].max()
    out['Number of Seasons Watched'] = df['Season'].nunique()
    if 'runtime' in df.columns:
        out['Total Time Watched (mins)'] = df['runtime'].sum()
        out['Total Time Watched (hrs)'] = np.round(out['Total Time Watched (mins)']/60, 2)
        out['Normal Episode Length'] = df['runtime'].value_counts().index[0]

    # Season Metrics
    season_gb = df.groupby('Season')
    season_episode_cnt = season_gb.size()
    time_to_watch_season = (season_gb['Date Watched'].max() - season_gb['Date Watched'].min())

    # Watch Rate is time/# of episodes
    episode_watch_rate = time_to_watch_season / season_gb['Date Watched'].size()

    # Longest Time to Watch
    out['Longest Time to Watch Season'] = time_to_watch_season.max()
    out['Season took Longest to Watch'] = time_to_watch_season.idxmax()
    out['# of Episode for Longest to Watch'] = season_episode_cnt.loc[time_to_watch_season.idxmax()]

    # Longest Days/Episode
    out['Longest Days/Episode'] = episode_watch_rate.max()
    out['Season with Longest Days/Episode'] = episode_watch_rate.idxmax()
    out['# of Episode for Longest Days/Episode'] = season_episode_cnt.loc[episode_watch_rate.idxmax()]

    # Shortest Time to Watch
    out['Shortest Time to Watch Season'] = time_to_watch_season.min()
    out['Season took Shortest to Watch'] = time_to_watch_season.idxmin()
    out['# of Episode for Shortest Time to Watch'] = season_episode_cnt.loc[time_to_watch_season.idxmin()]

    # Shortest Days/Episode
    out['Shortest Days/Episode'] = episode_watch_rate.min()
    out['Season with Shortest Days/Episode'] = episode_watch_rate.idxmin()
    out['# of Episode for Shortest Days/Episode'] = season_episode_cnt.loc[episode_watch_rate.idxmin()]

    out_series = pd.Series(out)
    return (out_series)


class DataPipeline():

    def __init__(self, netflix_csv_path, tmdb_key, sql_path, id=None):

        tmdb.API_KEY = tmdb_key

        self.search = tmdb.Search()
        self.netflix_path = netflix_csv_path
        self.con = sqlite3.connect(sql_path)
        self.date = datetime.datetime.now().date()
        if id == None:
            self.id = uuid.uuid1()
        else:
            self.id = id
        print(f"User: {self.id}")

    def execute(self):
        netflix_df = self.get_netflix_df(self.con, self.netflix_path, self.id)
        self.movie_df = self.generate_movie_df(netflix_df)
        self.tv_df = self.generate_tv_df(netflix_df)

    def get_netflix_df(self, con, netflix_path=None, id=None):
        tables = get_db_tables (con)
        if 'User_Shows' in tables:
            netflix_df = pd.read_sql(f"select * from User_Shows where User_ID = '{id}'", con=con)
            if netflix_df.shape[0] == 0:
                netflix_df = self.parse_netflix(netflix_path, user_id=self.id)
                netflix_df.to_sql('User_Shows', con, index=False, if_exists='append')

        else:
            netflix_df = self.parse_netflix(netflix_path, user_id=self.id)
            netflix_df.to_sql('User_Shows', con, index=False)

        return(netflix_df)

    def parse_netflix(self, data_path, show=True, user_id='NA'):

        """
        Parse the netflix dataframe to and whehter a row is a TV shows or Movie
        Creates data frame with columns ['Title', 'Date', 'Show Name', 'Season', 'Episode Name', 'TV Show flag']
        :param netflix_df:
        :param show:
        :return:
        """
        netflix_df = pd.read_csv(data_path)
        netflix_df['Date'] = pd.to_datetime(netflix_df['Date'])
        netflix_df.rename({'Date':'Date Watched'}, axis=1, inplace=True)

        # Split title into parts
        title_splits = netflix_df['Title'].str.split(':', 2, expand=True)
        title_splits.rename({0: 'Show Name', 1: 'Season', 2: 'Episode Name'}, axis=1, inplace=True)
        title_splits['Episode Name'] =  title_splits['Episode Name'].str.strip().tolist()
        # Call row a TV show  if episode name is full.
        title_splits['TV Show flag'] = title_splits['Episode Name'].apply(lambda row: 'Movie' if row == None else 'TV Show')
        # Combine and output
        self.netflix_df_full = pd.concat([netflix_df, title_splits], axis=1)
        self.netflix_df_full['User_ID'] = user_id
        self.netflix_df_full['Date Logged'] = str(datetime.datetime.now().date())

        if show: print(f"Total number of TV Show + Movies: {self.netflix_df_full.shape[0]}")

        return(self.netflix_df_full)

    @staticmethod
    def _get_best_tmdb_search_result(search_term, search_results):
        """
        Find best TMDB match based on name and popularity
        :return:
        """
        names = [{'New Title': x.get('title', x.get('name'))  # Movie API uses title, TV show uses name
                     , 'TMDB id': x['id']
                     , 'Popularity': x['popularity']
                     , 'Cosine Distance': fuzz.ratio(search_term, x.get('title', x.get('name')))} for x in search_results['results']]
        names_df = pd.DataFrame(names).sort_values(['Cosine Distance', 'Popularity'], ascending=False)
        keeper_id = names_df.iloc[0]['TMDB id']
        return(keeper_id)

    @staticmethod
    def clean_string(movie_name):
        cleaner_str = re.sub("[\(\[].*?[\)\]]", "", movie_name)
        cleaner_str = cleaner_str.split(':')[0]
        return(cleaner_str)

    def _get_movie_API_results(self, movie_title):
        """
        Query TMDB API for movie title, find best result, return selected fields
        :param movie_title:
        :return:
        """
        # Select requested fields from response
        normal_movie_fields = ['original_title' ,'budget', 'homepage', 'imdb_id', 'overview', 'popularity' \
            , 'release_date', 'revenue', 'runtime', 'vote_average', 'vote_count', 'tagline']

        # Find the Movie in TMDB
        search_results = self.search.movie(query=movie_title)
        n_results = len(search_results['results'])
        #     print("N Results: ", n_results)

        # Handle 0, 1, multiple tmdb search results
        if n_results == 0:

            # If no matched with raw input, try cleaning input and trying again
            cleaned_str = self.clean_string(movie_title)
            if cleaned_str != movie_title:

                movie_results2 = self._get_movie_API_results(cleaned_str)
                return(movie_results2)

            movie_results = {key: np.nan for key in normal_movie_fields}
            movie_results['Number of Search Results'] = n_results
            movie_results['Input Movie Title'] = movie_title
            movie_results['Cosine Distance'] = np.nan
            movie_results['Date Pulled'] = datetime.datetime.now().date()
            movie_results = pd.Series(movie_results)

            return (movie_results)

        elif n_results == 1:
            temp_id = search_results['results'][0]['id']
        elif n_results > 1:
            temp_id =  self._get_best_tmdb_search_result(movie_title, search_results)

        # Get TMDB attributes
        full_movie_results = tmdb.Movies(temp_id).info()
        movie_results = {attr: full_movie_results.get(attr) for attr in normal_movie_fields}

        # Genres
        if full_movie_results.get('genres') == None:
            movie_results['Genres'] = '[]'
            movie_results['Genre IDs'] = '[]'
        else:
            movie_results['Genres'] = str([x.get('name') for x in full_movie_results.get('genres')])
            movie_results['Genre IDs'] = str([x.get('id') for x in full_movie_results.get('genres')])

        # Append additional fields
        movie_results['Number of Search Results'] = n_results
        movie_results['Input Movie Title'] = movie_title
        movie_results['Cosine Distance'] = fuzz.ratio(movie_title, movie_results.get('original_title'))
        movie_results['Date Pulled'] = datetime.datetime.now().date()

        # Prepare to export
        movie_results = pd.Series(movie_results)
        new_column_order = movie_results.index.tolist().copy()
        new_column_order.remove('Input Movie Title')
        new_column_order.insert(0, 'Input Movie Title')
        movie_results = movie_results[new_column_order]

        return (movie_results)

    def generate_movie_df(self, netflix_df, show=True):
        """
        Create dataframe for movies

        Columns include ['Number of Search Results', 'budget', 'homepage', 'imdb_id', 'overview',
       'popularity', 'release_date', 'revenue', 'runtime', 'title_query',
       'vote_average', 'vote_count']


        :param netflix_df:
        :param pkl_path:
        :param show:
        :return:
        """

        user_movies = netflix_df[netflix_df['TV Show flag'] == 'Movie'][['Title', 'Date Watched']].copy()
        print(f'Total number of movies: {user_movies.shape[0]}')
        # Check DB for movies we already have info for...
        tables = get_db_tables (self.con)
        movie_table_exists = True
        if 'Movies' in tables:
            db_movies = pd.read_sql_query('select * from Movies', self.con)
            # TODO: Shouldn't need to dedup in prod db...
            db_movies = db_movies.drop_duplicates(subset=['Input Movie Title'])
            user_db_movies_merged = pd.merge(user_movies, db_movies
                                   , left_on='Title', right_on='Input Movie Title'
                                   , how='left')

            titles_found = user_db_movies_merged[user_db_movies_merged['original_title'].notnull()]
            print(f"Found {titles_found.shape[0]} matches in db ")

            titles_to_get = user_db_movies_merged[user_db_movies_merged['original_title'].isna()]['Title']
            print(f'Did not find {titles_to_get.shape[0]} titles, querying TMDB for results')
        else:
            movie_table_exists = False
            print("Movies table not in DB")
            titles_to_get = user_movies['Title']
            found_cols = ['original_title', 'budget', 'homepage', 'imdb_id', 'overview',
               'popularity', 'release_date', 'revenue', 'runtime', 'vote_average',
               'vote_count', 'tagline', 'Number of Search Results',
               'Input Movie Title', 'Cosine Distance', 'Date Pulled']
            titles_found = pd.DataFrame(columns=found_cols)

        # If there are no new title, then return merged db
        if titles_to_get.shape[0] == 0:
            self.all_movies_results_df = user_db_movies_merged.copy()
            return(self.all_movies_results_df)



        # Fresh Query
        new_movies_info = titles_to_get.progress_apply(self._get_movie_API_results)

        # Combine new movie info with db info
        self.all_movies_results_df = pd.concat([titles_found, new_movies_info])#.drop(['Title'], axis = 1)

        self.all_movies_results_df = pd.merge(self.all_movies_results_df, user_movies
                                              , left_on='Input Movie Title', right_on='Title'
                                              , how='inner')


        if show:
            print("Number of movies: ", self.all_movies_results_df.shape[0])
            missing_movies = self.all_movies_results_df[self.all_movies_results_df['original_title'].isna()]
            print("Number of movies not found in TMDB Search: ", missing_movies.shape[0])
            if missing_movies.shape[0] > 0: print(missing_movies['Input Movie Title'])

        # Add to Database
        new_movies_info = new_movies_info[new_movies_info['original_title'].notnull()]
        if new_movies_info.shape[0] > 0:
            if movie_table_exists:
                db = pd.read_sql_query('select original_title from Movies', self.con)
                keepers = pd.merge(new_movies_info, db, on='original_title'
                                   , how='left', indicator='i')

                keepers = keepers[keepers['i'] == 'left_only'].drop(['i'], axis=1)
                if show: print(f"Adding {keepers.shape[0]} to database")
                keepers.to_sql('Movies', self.con, index=False, if_exists='append')
            else: # Movies Table does not exist
                new_movies_info.to_sql('Movies', self.con, index=False, if_exists='append')


        return(self.all_movies_results_df)

    def _get_TV_show_details(self, tv_show_name):
        # Select requested_fields from response
        normal_tv_show_fields = ['n_production_companies', 'primary_production_co', 'runtime', 'release_date'
            , 'n_network', 'primary_network', 'homepage', 'overview', 'popularity', 'vote_average', 'vote_count'
            , 'in_production', 'type', 'status', 'number_of_seasons', 'number_of_episodes']

        # Search TMDB by TV show name
        search_results = self.search.tv(query=tv_show_name)
        n_results = len(search_results['results'])

        # Parse search response
        if n_results == 0:
            tv_show_results = {key:np.nan for key in normal_tv_show_fields}
            tv_show_results['Number of Search Results'] = n_results
            tv_show_results['Input TV Show Title'] = tv_show_name
            tv_show_results['Cosine Distance'] = np.nan
            tv_show_results['Date Pulled'] = datetime.datetime.now().date()
            tv_show_results = pd.Series(tv_show_results)
            return(tv_show_results)

        elif n_results == 1:
            temp_id = search_results['results'][0]['id']
        elif n_results > 1:
            temp_id = self._get_best_tmdb_search_result(tv_show_name, search_results)

        full_tv_show_results=tmdb.TV(temp_id).info()

        # Fill dictionary with response items directly
        normal_tv_fields = ['homepage', 'overview', 'popularity' \
            , 'vote_average', 'vote_count', 'name', 'id'
            , 'in_production', 'type', 'status', 'number_of_seasons', 'number_of_episodes']
        tv_show_results = {attr: full_tv_show_results.get(attr) for attr in normal_tv_fields}

        # Attributes that require some preprocessing
        tv_show_results['n_production_companies'] = len(full_tv_show_results['production_companies'])
        tv_show_results['primary_production_co'] = '' if len(full_tv_show_results['production_companies']) == 0 else \
        full_tv_show_results['production_companies'][0]['name']
        tv_show_results['runtime'] = 0 if len(full_tv_show_results['episode_run_time']) == 0 else \
            full_tv_show_results['episode_run_time'][0]
        tv_show_results['release_date'] = full_tv_show_results['first_air_date']
        tv_show_results['n_network'] = len(full_tv_show_results['networks'])
        tv_show_results['primary_network'] = '' if len(full_tv_show_results['networks']) == 0 else \
        full_tv_show_results['networks'][0]['name']

        # Genres
        if full_tv_show_results.get('genres') == None:
            tv_show_results['Genres'] = '[]'
            tv_show_results['Genre IDs'] = '[]'
        else:
            tv_show_results['Genres'] = str([x.get('name') for x in full_tv_show_results.get('genres')])
            tv_show_results['Genre IDs'] = str([x.get('id') for x in full_tv_show_results.get('genres')])


        # Append additional fields
        tv_show_results['Number of Search Results'] = n_results
        tv_show_results['Input TV Show Title'] = tv_show_name
        tv_show_results['Cosine Distance'] = fuzz.ratio(tv_show_name, tv_show_results.get('original_name'))
        tv_show_results['Date Pulled'] = datetime.datetime.now().date()

        tv_show_results = pd.Series(tv_show_results)
        return (tv_show_results)

    def generate_tv_df(self, netflix_df, show=True):

        all_tv_episodes = netflix_df[netflix_df['TV Show flag'] == 'TV Show']
        print(f"Total number of Episodes of TV Shows: {all_tv_episodes.shape[0]}")
        user_tv_shows = all_tv_episodes.groupby('Show Name').apply(get_series_watched_gb)
        user_tv_shows = user_tv_shows.reset_index()
        if show: print(f'Number of unique TV Shows: {user_tv_shows.shape[0]}')
        # Check DB for movies we already have info for...
        tables = get_db_tables (self.con)
        tv_show_table_exits = True
        if 'TV_Shows' in tables:
            db_tv_shows = pd.read_sql_query('select * from TV_Shows', self.con)
            # TODO: Shouldn't need to dedup in prod db...
            db_tv_shows = db_tv_shows.drop_duplicates(subset=['Input TV Show Title'])
            user_db_movies_merged = pd.merge(user_tv_shows, db_tv_shows
                                             , left_on='Show Name', right_on='Input TV Show Title'
                                             , how='left')

            titles_found = user_db_movies_merged[user_db_movies_merged['Input TV Show Title'].notnull()]
            print(f"Found {titles_found.shape[0]} matches in db ")

            titles_to_get = user_db_movies_merged[user_db_movies_merged['Input TV Show Title'].isna()]['Show Name']
            print(f'Did not find {titles_to_get.shape[0]} titles, querying TMDB for results')
        else:
            tv_show_table_exits = False
            print("TV Shows table not in DB ")
            titles_to_get = user_tv_shows['Show Name']
            tv_show_cols = ['homepage',
             'overview',
             'popularity',
             'vote_average',
             'vote_count',
             'name',
             'id',
             'in_production',
             'type',
             'status',
             'number_of_seasons',
             'number_of_episodes',
             'n_production_companies',
             'primary_production_co',
             'runtime',
             'release_date',
             'n_network',
             'primary_network',
             'Number of Search Results',
             'Input TV Show Title',
             'Cosine Distance',
             'Date Pulled']
            titles_found = pd.DataFrame(columns=tv_show_cols)

        # If there are no new title, then return merged db
        if titles_to_get.shape[0] == 0:
            self.all_tv_shows_results_df = user_db_movies_merged.copy()
            return(self.all_tv_shows_results_df)


        # Fresh Query
        # titles_to_get = titles_to_get.iloc[:20] # Testing
        new_tv_shows_info = titles_to_get.progress_apply(self._get_TV_show_details)

        # Combine new movie info with db info
        self.all_tv_shows_results_df = pd.concat([titles_found, new_tv_shows_info])

        self.all_tv_shows_results_df = pd.merge(self.all_tv_shows_results_df, user_tv_shows
                                              , left_on='Input TV Show Title', right_on='Show Name'
                                              , how='inner')

        if show:
            missing_movies = self.all_tv_shows_results_df[self.all_tv_shows_results_df['name'].isna()]
            print("Number of tv shows not found in TMDB Search: ", missing_movies.shape[0])
            if missing_movies.shape[0] > 0: print(missing_movies['Input TV Show Title'])

        # Add to Database
        print(new_tv_shows_info.columns)
        new_tv_shows_info = new_tv_shows_info[new_tv_shows_info['Input TV Show Title'].notnull()]
        if new_tv_shows_info.shape[0] > 0:
            if tv_show_table_exits:

                # Add only the new entries (special because of re search feature may not have original name in databsec
                db_tv_show = pd.read_sql_query('Select `Input TV Show Title` from TV_Shows', self.con)
                keepers = pd.merge(new_tv_shows_info, db_tv_show, on='Input TV Show Title'
                                   , how='left', indicator='i')

                keepers = keepers[keepers['i'] == 'left_only'].drop(['i'], axis=1)
                if show: print(f"Adding {keepers.shape[0]} to database")
                keepers.to_sql('TV_Shows', self.con, index=False, if_exists='append')
            else:  # Movies Table does not exist
                new_tv_shows_info.to_sql('TV_Shows', self.con, index=False, if_exists='append')

        return(self.all_tv_shows_results_df)



if __name__ =='__main__':
    netflix_path = '../Data/NetflixViewingHistory.csv'
    db_location = '../Data/netflix_viewing1.db'
    with open('../Data/api_key.pkl', 'rb') as hnd:
        key = pickle.load(hnd)['api_key']

    data_pipeline = DataPipeline(netflix_path, key, db_location)

    netflix_df = data_pipeline.parse_netflix(netflix_path)
    movie_df = data_pipeline.generate_movie_df(netflix_df)
    tv_df = data_pipeline.generate_tv_df(netflix_df)


