import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession, Window
import pyspark.sql.types as t
import pyspark.sql.functions as f

spark_session = (SparkSession.builder \
    .master("local") \
    .appName("task app") \
    .config(conf=SparkConf()) \
    .getOrCreate())

people_info_schema = t.StructType([
    t.StructField("Id", t.StringType(), False),
    t.StructField("Name", t.StringType(), False),
    t.StructField("Birth year", t.IntegerType(), False),
    t.StructField("Death Year", t.IntegerType(), True),
    t.StructField("Professions", t.StringType(), True),
    t.StructField("Titles", t.StringType(), True)
    ])

title_akas_schema = t.StructType([
    t.StructField("Title Id", t.StringType(), False),
    t.StructField("Ordering", t.IntegerType(), False),
    t.StructField("Title", t.StringType(), False),
    t.StructField("Region", t.StringType(), True),
    t.StructField("Language", t.StringType(), True),
    t.StructField("Types", t.StringType(), True),
    t.StructField("Attributes", t.StringType(), True),
    t.StructField("IsOriginalTitle", t.BooleanType(), True),
])

title_crew_schema = t.StructType([
    t.StructField("Title Id", t.StringType(), False),
    t.StructField("Directors", t.StringType(), True),
    t.StructField("Writers", t.StringType(), True)
])

title_principals_schema = t.StructType([
    t.StructField("Title Id", t.StringType(), False),
    t.StructField("Ordering", t.IntegerType(), False),
    t.StructField("Person Id", t.StringType(), False),
    t.StructField("Category", t.StringType(), False),
    t.StructField("Job", t.StringType(), True),
    t.StructField("Characters", t.StringType(), True),
])

title_basic_schema = t.StructType([
    t.StructField("Title Id", t.StringType(), False),
    t.StructField("Title Type", t.StringType(), False),
    t.StructField("Primary Title", t.StringType(), False),
    t.StructField("Original Title", t.StringType(), False),
    t.StructField("Is Adult", t.IntegerType(), False),
    t.StructField("Start Year", t.IntegerType(), False),
    t.StructField("End Year", t.IntegerType(), True),
    t.StructField("Runtime Minutes", t.IntegerType(), True),
    t.StructField("Genres", t.StringType(), True)
])

title_episode_schema = t.StructType([
    t.StructField("Episode Id", t.StringType(), False),
    t.StructField("Title Id", t.StringType(), False),
    t.StructField("Season Number", t.IntegerType(), True),
    t.StructField("Episode Number", t.IntegerType(), True)
])

title_ratings_schema = t.StructType([
    t.StructField("Title Id", t.StringType(), False),
    t.StructField("Average Rating", t.FloatType(), False),
    t.StructField("Votes Number", t.IntegerType(), False)
])

def load_dataset(schema, filename, show=False, path='../dataset/'):
    df = spark_session.read.csv(path + filename, sep='\t', header=True, schema=schema, nullValue='\\N')
    if show:
        df.show()
    return df

people_info_df = load_dataset(people_info_schema, 'name_basic.tsv')
title_akas_df = load_dataset(title_akas_schema, 'title_akas.tsv')
title_crew_df = load_dataset(title_crew_schema, 'title_crew.tsv')
title_principals_df = load_dataset(title_principals_schema, 'title_principals.tsv')
title_basic_df = load_dataset(title_basic_schema, 'title_basic.tsv')
title_episode_df = load_dataset(title_episode_schema, 'title_episode.tsv')
title_ratings_df = load_dataset(title_ratings_schema, 'title_ratings.tsv')

def get_ukr_titles(df):
    sub_df = df.filter(f.col('Region') == 'UA').select('title')
    sub_df.write.csv('ukr_titles.csv', header=True, mode='overwrite')

def get_19th_century_born(df):
    sub_df = df.filter((f.col('Birth year') >= 1800) & (f.col('Birth year') < 1900)) \
        .select('Name')
    sub_df.write.csv('19th_century_born.csv', header=True, mode='overwrite')

def get_long_movies(df):
    sub_df = df.filter((f.col('Title Type') == 'movie') & (f.col('Runtime Minutes') > 120)) \
        .select('Original Title')
    sub_df.write.csv('long_movies.csv', header=True, mode='overwrite')

def get_actors_info(title_principals_df, people_info_df, title_basic_df):
    sub_df = title_principals_df.join(people_info_df, title_principals_df['Person Id'] == people_info_df['Id'], "inner") \
        .filter(f.col('Category').isin(['actor', 'actress'])) \
        .select('Title Id', 'Name', 'Characters') \
        .join(title_basic_df, on="Title Id") \
        .select('Name', 'Original Title', 'Characters')
    sub_df.write.csv('actors_info.csv', header=True, mode='overwrite')

def get_titles_per_region(title_basic_df, title_akas_df):
    sub_df = title_basic_df.join(title_akas_df, "Title Id", "inner") \
        .select('Title Id', 'Region') \
        .groupBy('Region').count().orderBy('count', ascending=False).limit(100)
    sub_df.write.csv('titles_per_region.csv', header=True, mode='overwrite')

def get_episodes_in_series(title_episode_df, title_basic_df):
    sub_df = title_episode_df.groupBy('Title Id').count() \
        .join(title_basic_df, 'Title Id', "inner") \
        .orderBy('count', ascending=False).limit(50) \
        .select('Original Title', 'count')
    sub_df.write.csv('episodes_in_series.csv', header=True, mode='overwrite')

def get_most_popular_per_decade(title_basic_df, title_ratings_df):
    sub_df = title_basic_df.na.drop(subset=['Start Year']).withColumn('decade_start', 
        f.floor(f.col('Start Year') / 10) * 10) 
    sub_df = sub_df.withColumn('decade_end', 
        f.col('decade_start') + 9)
    sub_df = sub_df.withColumn('decade', 
        f.concat_ws('-', f.col('decade_start').cast('string'), f.col('decade_end').cast('string')))
    sub_df = sub_df.drop('decade_start', 'decade_end')
    sub_df = sub_df.join(title_ratings_df,'Title Id', 'inner')
    window = Window.orderBy('Average Rating').partitionBy('decade')
    sub_df = sub_df.withColumn('Row number', f.row_number().over(window)) \
        .filter(f.col('Row number') <= 10) \
        .select('Original Title', 'decade')
    sub_df.write.csv('most_popular_per_decade.csv', header=True, mode='overwrite')

def get_most_popular_by_genre(title_basic_df, title_ratings_df):
    sub_df = title_basic_df.na.drop(subset=['Genres'])\
        .join(title_ratings_df, 'Title Id', 'inner')
    window = Window.orderBy('Average Rating').partitionBy('Genres')
    sub_df = sub_df.withColumn('Row number', f.row_number().over(window)) \
        .filter(f.col('Row number') <= 10) \
        .select('Original Title', 'Genres')
    sub_df.write.csv('most_popular_by_genre.csv', header=True, mode='overwrite')

get_ukr_titles(title_akas_df)
get_19th_century_born(people_info_df)
get_long_movies(title_basic_df)
get_actors_info(title_principals_df, people_info_df, title_basic_df)
get_titles_per_region(title_basic_df, title_akas_df)
get_episodes_in_series(title_episode_df, title_basic_df)
get_most_popular_per_decade(title_basic_df, title_ratings_df)
get_most_popular_by_genre(title_basic_df, title_ratings_df)