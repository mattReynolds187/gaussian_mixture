import pandas as pd
import numpy as np
import argparse
import os

"""
This is a script for preparing the data downloaded from https://archive.org/download/nf_prize_dataset.tar
to be used by the GMM class (generates an X array where each row corresponds to a user and each column
corresponds to a movie). The output matrix X is stored in a text file 'my_array.txt'
The training_set folder contains over 17000 text files for (one for each movie). Each line in each movie file is a
rating for that movie from a specific user. The first two arguments for this script let you control how
many movies are used in the array and how many reviews to take from each movie (hence it's customizable
to your computers memory capabilities)

usage from command line:
first argument: number of movies to include
second argument: number of reviews to take from each movie file
third argument: save location of generated array

This script expects the extracted 'training_set' folder to be in the same directory as this script.
"""

parser = argparse.ArgumentParser()
parser.add_argument('movie_num', type=int)
parser.add_argument('lines_per_movie', type=int)
parser.add_argument('save_location', type=str)
args = parser.parse_args()
file_names = os.listdir('./training_set')[:args.movie_num]
lines_per_movie_file = args.lines_per_movie
save_location = args.save_location

def generate_big_csv(csv_name):
    with open(csv_name, 'w+') as output_file:
        output_file.write("user_id,rating,date,movie_id" + '\n')
        counter = -1
        for file_name in file_names:
            counter += 1
            if counter % 100 == 0:
                print("Number of movies parsed: ", counter)
            if file_name.endswith(".txt"):
                with open(os.path.join(os.getcwd(), 'training_set', file_name)) as f:
                    to_append = "," + f.readline()[:-2]
                    counter = 0
                    for line in f:
                        counter += 1
                        output_file.write(line[:-1] + to_append + '\n')
                        if counter == lines_per_movie_file:
                            break;

generate_big_csv('movies.csv')

movies = pd.read_csv('movies.csv')
unique_users = movies.user_id.unique()
unique_movie_ids = movies.movie_id.unique()
del movies
user_to_index = {}
movie_to_index = {}

print("Building user to index dictionary...\n")
for i, user_id in enumerate(unique_users):
    user_to_index[user_id] = i

print("Building movie to index dictionary...\n")
for i, movie_id in enumerate(unique_movie_ids):
    movie_to_index[movie_id] = i

num_users = len(unique_users)
num_movies = len(unique_movie_ids)
del unique_users
del unique_movie_ids

print("Building array for", num_users, "users and", num_movies, "movies...\n")
try:
    X = np.zeros([num_users, num_movies], dtype=np.int8)
except MemoryError:
    print('Not enough memory!')

with open("movies.csv") as f:
    f.readline()
    for line in f:
        splits = line.split(',')
        rating = int(splits[1])
        row_num = user_to_index[int(splits[0])]
        column_num = movie_to_index[int(splits[3])]
        X[row_num][column_num] = rating

print("Saving array...\n")
np.savetxt(save_location, X, '%d')
print("Done.\n")
