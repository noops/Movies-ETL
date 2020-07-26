# Movies-ETL
## Resources: 
data: movies_metadata.csv, ratings.csv, wikipedia.moves.json

software: postgresql, pgadmin 4, python, visual studio code

## Problem:
Create a python script capable of transforming and loading data from multiple sources into a database. Challenge.py has a function that accepts data from a json file and two csv files. It cleans the data to our specifications and then loads it into a postgresql database. I attempted to use code found on stackoverflow to remove existing data from SQL while keeping the empty tables, but was unable to get it to run with my code. An if_exists = ‘append’ seemed to fix this problem. I moved the clean movies function outside of my main transform and merge function for ease of reading and debugging. The ratings file is large and took over an hour to load into SQL. It contains over 26 million rows.

## Takeaways:
During out data exploration in ETL.ipynb we created scatterplots to analyze wikipedia data vs kaggle data. The kaggle data was more consistent so those columns were favored over the wikipedia columns. Columns that are dropped completely were assumed to have little relevance to the rest of the data. The dropped columns contained greater than 90% null values.  
