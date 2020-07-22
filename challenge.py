#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import re
import time

from sqlalchemy import create_engine
from config import db_password

file_dir = '/Users/admin/Desktop/Data_Analytics_BootCamp/Challenge8_Movies-ETL'


with open(f'{file_dir}/wikipedia.movies.json', mode='r') as file:
    wiki_movies_raw = json.load(file)

kaggle_metadata = pd.read_csv(f'{file_dir}/movies_metadata.csv', low_memory=False)
ratings = pd.read_csv(f'{file_dir}/ratings.csv')

wiki_movies_df = pd.DataFrame(wiki_movies_raw)

wiki_movies_df.columns.tolist()

# using list comprehensions for filter data
wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie 
                   and 'No. of episodes' not in movie]

def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    alt_titles = {}
    for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune–Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
        if key in movie:
            alt_titles[key] = movie[key]
            movie.pop(key)
    if len(alt_titles) > 0:
        movie['alt_titles'] = alt_titles
    # merge column names
    def change_column_name(old_name, new_name):
        if old_name in movie:
            movie[new_name] = movie.pop(old_name)
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

clean_movies = [clean_movie(movie) for movie in wiki_movies]

wiki_movies_df[wiki_movies_df['Arabic'].notnull()]['url']

wiki_movies_df = pd.DataFrame(clean_movies)
sorted(wiki_movies_df.columns.tolist())

clean_movies = [clean_movie(movie) for movie in wiki_movies]
wiki_movies_df = pd.DataFrame(clean_movies)

#Removing duplicated data
wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
print(len(wiki_movies_df))
wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
print(len(wiki_movies_df))

# finding the null columns
null_values = [[column,wiki_movies_df[column].isnull().sum()] for column in wiki_movies_df.columns]

# Remove mostly null columns
wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]
wiki_movies_df.columns

# make a data series that drops missing values
box_office = wiki_movies_df['Box office'].dropna() 

def is_not_a_string(x):
    return type(x) != str

box_office[box_office.map(lambda x: type(x) != str)]

box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)

form_one = r'\$\d+\.?\d*\s*[mb]illi?on'
box_office.str.contains(form_one, flags=re.IGNORECASE).sum()

form_two = r'\$\d{1,3}(?:,\d{3})+'
box_office.str.contains(form_two, flags=re.IGNORECASE).sum()

matches_form_one = box_office.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = box_office.str.contains(form_two, flags=re.IGNORECASE)

box_office[~matches_form_one & ~matches_form_two]

form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'
box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)
box_office.str.extract(f'({form_one}|{form_two})')

def parse_dollars(s):
    # if s is not a string, return NaN
    if type(s) != str:
        return np.nan

    # if input is of the form $###.# million
    if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " million"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a million
        value = float(s) * 10**6

        # return value
        return value

    # if input is of the form $###.# billion
    elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " billion"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a billion
        value = float(s) * 10**9

        # return value
        return value

    # if input is of the form $###,###,###
    elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

        # remove dollar sign and commas
        s = re.sub('\$|,','', s)

        # convert to float
        value = float(s)

        # return value
        return value

    # otherwise, return NaN
    else:
        return np.nan

# extract the values from box_office using str.extract
wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', 
                                                      flags=re.IGNORECASE)[0].apply(parse_dollars)

wiki_movies_df.drop('Box office', axis=1, inplace=True)

budget = wiki_movies_df['Budget']

# Parse Budget Data
budget = wiki_movies_df['Budget'].dropna()
budget

# Convert any lists to strings
budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)

# remove any values between a dollar sign and a hyphen 
budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

# Parse box budget data
matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)
budget = budget.str.replace(r'\[\d+\]\s*', '')
budget[~matches_form_one & ~matches_form_two]

wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)

wiki_movies_df.drop('Budget', axis=1, inplace=True)

# Parse Release Date
release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
date_form_two = r'\d{4}.[01]\d.[123]\d'
date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
date_form_four = r'\d{4}'


wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)

# Parse Running Time
running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)

running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE).sum()

running_time[running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE) != True]

running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE).sum()

running_time[running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE) != True]

running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')

running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)

wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 
                                                            else row[2], 
                                                            axis=1)

wiki_movies_df.drop('Running time', axis=1, inplace=True)

kaggle_metadata['adult'].value_counts()

# Remove Bad Data
kaggle_metadata[~kaggle_metadata['adult'].isin(['True','False'])]

kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')

kaggle_metadata['video'].value_counts()

kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'

kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')

kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])

# rating data
ratings.info(null_counts=True)

pd.to_datetime(ratings['timestamp'], unit='s')

ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

ratings['rating'].plot(kind='hist')
ratings['rating'].describe()

# merge Wikipedia and Kaggle Metadata
movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])

# merge title
movies_df[(movies_df['title_kaggle'] == '') | (movies_df['title_kaggle'].isnull())]

# merge running_time
movies_df.fillna(0).plot(x='running_time', y='runtime', kind='scatter')

# budget
movies_df.fillna(0).plot(x='budget_wiki',y='budget_kaggle', kind='scatter')

# Box Office
movies_df.fillna(0)[movies_df['box_office'] < 10**9].plot(x='box_office', y='revenue', kind='scatter')

# Release Date
movies_df[['release_date_wiki','release_date_kaggle']].plot(x='release_date_wiki', y='release_date_kaggle', style='.')

movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')]

movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') 
                                     & (movies_df['release_date_kaggle'] < '1965-01-01')].index)

movies_df[movies_df['release_date_wiki'].isnull()]

# Language
movies_df['Language'].apply(lambda x: tuple(x) if type(x) == list else x).value_counts(dropna=False)

movies_df['original_language'].value_counts(dropna=False)

movies_df[['Production company(s)','production_companies']]

# drop the title_wiki, release_date_wiki, Language, and Production company(s) columns
movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)

def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
    df[kaggle_column] = df.apply(
        lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
        , axis=1)
    df.drop(columns=wiki_column, inplace=True)

fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')
movies_df

# check the merged data if there's any columns with only one value. 
for col in movies_df.columns:
    lists_to_tuples = lambda x: tuple(x) if type(x) == list else x
    value_counts = movies_df[col].apply(lists_to_tuples).value_counts(dropna=False)
    num_values = len(value_counts)
    if num_values == 1:
        print(col)

movies_df['video'].value_counts(dropna=False)

# reorder the columns
movies_df = movies_df.loc[:, ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection',
                              'url','imdb_link','runtime','budget_kaggle','revenue','release_date_kaggle',
                              'popularity','vote_average','vote_count','genres','original_language','overview',
                              'spoken_languages','Country','production_companies','production_countries',
                              'Distributor','Producer(s)','Director','Starring','Cinematography','Editor(s)',
                              'Writer(s)','Composer(s)','Based on']
                         ]

# rename the columns
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

# creating pivot that movieId becomes the index
rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count().rename({'userId':'count'}, axis=1)               .pivot(index='movieId',columns='rating', values='count')

# prepend rating_ to each column with a list comprehension
rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]

# using left merge/join for movie_df and rating_counts
movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')

# fill-in 0 for movies that got zero rating
movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)

# connection to local server
db_string = f"postgres://postgres:{db_password}@127.0.0.1:5433/movie_data_challenge"

# create the database engine
engine = create_engine(db_string)

# import movie data
try: 
    movies_df.to_sql(name='movies', con=engine)

except Exception as ex:
    print(type(ex).__name__)
    print("Please Check the Datasets Again")

# import the Ratings Data
rows_imported = 0

# get the start_time from time.time()
start_time = time.time()

for data in pd.read_csv(f'{file_dir}/ratings.csv', chunksize=1000000):

    # print out the range of rows that are being imported
    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
    
    data.to_sql(name='ratings', con=engine, if_exists='append')

    # increment the number of rows imported by the chunksize
    rows_imported += len(data)
    
    # print that the rows have finished importing
    print(f'Done. {time.time() - start_time} total seconds elapsed')
