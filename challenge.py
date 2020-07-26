import pandas as pd
import numpy as np
import re
import json
import psycopg2
#import contextlib
#from sqlalchemy import MetaData
from sqlalchemy import create_engine
from config import db_password


#global variables
file_dir = '/Users/bkirton/desktop/dataAustin2020/Movies-ETL/resources/'
wiki_data = f'{file_dir}wikipedia.movies.json'
kaggle_data = f'{file_dir}movies_metadata.csv'
movie_ratings = f'{file_dir}ratings.csv'



#function accepts 3 files to transform and merge
def transform_and_merge(wiki_data, kaggle_data, movie_ratings):
    
    #load wiki data into a dataframe
    with open(wiki_data, mode='r') as file:
        wiki_movies_raw = json.load(file)
    wiki_movies_df = pd.DataFrame(wiki_movies_raw)

    wiki_movies = [movie for movie in wiki_movies_raw 
                if ('Director' in movie or 'Directed by' in movie) 
                and 'imdb_link' in movie
                and 'No. of episodes' not in movie]

    #loop through data to get clean movies and load into data frame
    clean_movies = [clean_movie(movie) for movie in wiki_movies]
    wiki_movies_df = pd.DataFrame(clean_movies)     

    #import kaggle and rating data into separate dataframes
    kaggle_metadata = pd.read_csv(kaggle_data)
    ratings = pd.read_csv(movie_ratings)

    #regex string extraction for imdb links and creation of imdb column
    wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')

    #drop duplicates 
    wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)

    #find number of null values in each column
    [[column,wiki_movies_df[column].isnull().sum()] for column in wiki_movies_df.columns]

    #remove columns with most null values
    wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
    wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]

    #drop null value rows in box office column
    box_office = wiki_movies_df['Box office'].dropna()
    #lists to strings 
    box_office[box_office.map(lambda x: type(x) != str)]

    #regex strings to find specifically formatted numbers 
    form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
    form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'

    def parse_dollars(s):
        # if s is not a string return NaN
        if type(s) != str:
            return np.nan
        # if input is of the form $###.# million
        if re.match(r'\$\s*\d+\.?\d*\s*milli?on',s,flags=re.IGNORECASE):
            # remove dollar sign and "million"
            s = re.sub('\$|\s|[a-zA-Z]','',s)
            # convert to float and multiply by a million
            value = float(s)*10**6
            # return value
            return value
        # if input is of the form $###.# billion
        elif re.match(r'\$\s*\d+\.?\d*\s*billi?on',s,flags=re.IGNORECASE):
            # remove dollar sign and "billion"
            s = re.sub('\$|\s|[a-zA-Z]','',s)
            # convert to float and multiply by a billion
            value = float(s)*10**9
            # return value
            return value
        # if input is of the form $###,###,###
        elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)',s,flags=re.IGNORECASE):
            # remove dollar sign and commas
            s = re.sub('\$|,','',s)
            # convert to float
            value = float(s)
            # return value
            return value
        # otherwise return NaN
        else:
            return np.nan


    # extract values from box_office and apply parse_dollars
    wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
    # drop box office column
    wiki_movies_df.drop('Box office', axis=1, inplace=True)

    #clean budget data
    #create budget variable drop null values
    budget = wiki_movies_df['Budget'].dropna()
    #convert any list to strings
    budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)
    #remove whitespaces
    box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)
    budget = budget.str.replace(r'\[\d+\]\s*', '')
    #extract data from budget and apply parse dollars. drop budget column
    wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
    wiki_movies_df.drop('Budget', axis=1, inplace=True)

    #clean release data
    # convert lists to strings for release date
    release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
    date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
    date_form_two = r'\d{4}.[01]\d.[123]\d'
    date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
    date_form_four = r'\d{4}'
    #extract dates
    release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})', flags=re.IGNORECASE)
    #convert to datetime using pandas
    wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)

    #clean running time data
    #convert lists to strings
    running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
    #extract values with new pattern
    running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')
    #convert strings to numeric values
    running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)
    #save output to dataframe
    wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)
    #drop running time column
    wiki_movies_df.drop('Running time', axis=1, inplace=True)

    #clean kaggle data
    #drop adult column
    kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')
    kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'
    kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
    kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
    kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')
    kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])

    #clean ratings data
    pd.to_datetime(ratings['timestamp'], unit='s')
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

    #merge wiki movies and kaggle 
    movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki', '_kaggle'])

    #drop columns
    movies_df.drop(columns=['title_wiki', 'release_date_wiki', 'Language','Production company(s)'], inplace=True)


    #fills in missing data for a column pair and drops redundant column
    def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
        df[kaggle_column] = df.apply(
            lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column], axis=1
        )
        df.drop(columns= wiki_column, inplace=True)

    #use function for 3 columns we decided to set to 0 (running_time, budget_wiki, box_office)
    fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
    fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
    fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')

    #double check that no columns contain only 1 value, convert lists to tuples for value_counts() to work
    for col in movies_df.columns:
        lists_to_tuples = lambda x: tuple(x) if type(x) == list else x
        value_counts = movies_df[col].apply(lists_to_tuples).value_counts(dropna=False)
        num_values = len(value_counts)
        if num_values == 1:
            print(col)

    #reorder columns for ease of reading 
    movies_df = movies_df.loc[:, ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                        'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                        'genres','original_language','overview','spoken_languages','Country',
                        'production_companies','production_countries','Distributor',
                        'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                        ]]
    movies_df.rename({'id':'kaggle_id',
                    'title_kaggle':'title',
                    'url':'wikipedia_url',
                    'budget_kaggle':'budget',
                    'release_date_kaggle':'release_date',
                    'Country':'country',
                    'Distributor':'distributor',
                    'Producer(s)':'producers',
                    'Director':'director',
                    'Starring':'starring',
                    'Cinematography':'cinematography',
                    'Editor(s)':'editors',
                    'Writer(s)':'writers',
                    'Composer(s)':'composers',
                    'Based on':'based_on'
                    }, axis='columns', inplace=True)

    #include raw ratings data and rename userId to count and pivot data so that movieId becomes the index
    rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count() \
                    .rename({'userId':'count'}, axis=1) \
                    .pivot(index='movieId', columns='rating', values='count')

    # rename columns for ease of understanding. prepend 'rating_' to each column
    rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]

    #left merge to keep all data in movies_df
    movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')

    #deal with missing values
    movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)

    #sql connection
    db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/movie_data"
    engine = create_engine(db_string)
    #clear database but keep existing tables https://stackoverflow.com/questions/4763472/sqlalchemy-clear-database-content-but-dont-drop-the-schema
    #meta = MetaData()
    #with contextlib.closing(engine.connect()) as con:
    #    trans = con.begin()
    #for table in reversed(meta.sorted_tables):
    #    con.execute(table.delete())
    #trans.commit()
    # save movies_df to sql table
    movies_df.to_sql(name='movies', con=engine, if_exists='append')
    #load ratings in chunks due to large file size 
    for data in pd.read_csv(f'{file_dir}ratings.csv', chunksize=1000000):
        data.to_sql(name='ratings', con=engine, if_exists='append')

def clean_movie(movie):
    #create a non destructive copy
    movie = dict(movie)
    alt_titles = {}
    #combine alternate titles into one list
    for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune–Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
        if key in movie:
            alt_titles[key] = movie[key]
            movie.pop(key)
    if len(alt_titles)>0:
        movie['alt_titles'] = alt_titles
    #merge column names
    def change_column_name(old_name, new_name):
        if old_name in movie:
            movie[new_name] = movie.pop(old_name)
    #change column names 
    change_column_name('Adaptation by', 'Writer(s)')
    change_column_name('Country of origin', 'Country')
    change_column_name('Directed by', 'Director')
    change_column_name('Distributed by', 'Distributor')
    change_column_name('Edited by', 'Editor(s)')
    change_column_name('Length', 'Running time')
    change_column_name('Original release', 'Release date')
    change_column_name('Music by', 'Composer(s)')
    change_column_name('Produced by', 'Producer(s)')
    change_column_name('Producer', 'Producer(s)')
    change_column_name('Productioncompanies ', 'Production company(s)')
    change_column_name('Productioncompany ', 'Production company(s)')
    change_column_name('Released', 'Release Date')
    change_column_name('Release Date', 'Release date')
    change_column_name('Screen story by', 'Writer(s)')
    change_column_name('Screenplay by', 'Writer(s)')
    change_column_name('Story by', 'Writer(s)')
    change_column_name('Theme music composer', 'Composer(s)')
    change_column_name('Written by', 'Writer(s)')

    return movie

#call function to pass in dataframes 
transform_and_merge(wiki_data, kaggle_data, movie_ratings)
