################################################################################
#
# File: project3
# Author: Michael Bechtel
# Date: September 28, 2020
# Class: EECS 731
# Description: Use slustering models to find similar movies in a given
#               movie list using features such as the genres and ratings.
# 
################################################################################

# Python imports
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

# Create the clustering models
kmeans_model = KMeans(n_clusters=2, random_state=0)
gmm_model = GaussianMixture(n_components=2)
ms_model = MeanShift()
dbscan_model = DBSCAN()
hac_model = AgglomerativeClustering(n_clusters=2)

# Read the raw datasets
movie_list = pd.read_csv("../data/raw/movies.csv")
ratings_list = pd.read_csv("../data/raw/ratings.csv")

# Create a 2D array / matrix for holding the individual genres for each movie
#   Each cell represents a movie and genre combination
#   If the movie is categorized as a genre, the cell will be set to a 1 value
#   Otherwise, the cell will be set to a 0 value
genre_list = ["Action","Adventure","Animation","Children","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"]
genre_matrix = []
for genre in genre_list:
    genre_matrix.append([])

# Parse through the movies. The following features are obtained:
#   -Movie IDs
#   -Average movie rating (get all rating values for a movie and compute the mean)
#   -Movie genres (fill the genre_matrix based on the criteria given above
movie_ids = []
movie_genres = []
movie_ratings = []
for i in range(len(movie_list)):
    movie_ids.append(movie_list["movieId"][i])
    movie_ratings.append(ratings_list.query("movieId=={}".format(movie_ids[i]))["rating"].mean())
    movie_genres.append(movie_list["genres"][i].split("|"))
    for j,genre in enumerate(genre_list):
        if genre in movie_genres[i]:
            genre_matrix[j].append(1)
        else:
            genre_matrix[j].append(0)    
    
# Create a new dataset with the obtained movie features
#   Save the new dataset to the data/processed/ directory
movies_dataset = pd.DataFrame({"movieId":movie_ids,"rating":movie_ratings,
                               "Action":genre_matrix[0],"Adventure":genre_matrix[1],
                               "Animation":genre_matrix[2],"Children":genre_matrix[3],
                               "Comedy":genre_matrix[4],"Crime":genre_matrix[5],
                               "Documentary":genre_matrix[6],"Drama":genre_matrix[7],
                               "Fantasy":genre_matrix[8],"Film-Noir":genre_matrix[9],
                               "Horror":genre_matrix[10],"Musical":genre_matrix[11],
                               "Mystery":genre_matrix[12],"Romance":genre_matrix[13],
                               "Sci-Fi":genre_matrix[14],"Thriller":genre_matrix[15],
                               "War":genre_matrix[16],"Western":genre_matrix[17]}).dropna()
movies_dataset.to_csv("../data/processed/movies_dataset.csv")      

# For each genre
for genre in genre_list:
    # Print the current genre
    print("Performing clustering on == {}".format(genre))

    # Get the rating and genre columns from the overall dataset
    genre_dataset = movies_dataset.loc[:,("rating",genre)]

    # Create a grid of graphs so all results can be saved to a single image
    #   Only 5 models are tested, so the sixth graph is removed
    _,graphs = plt.subplots(2,3)  
    graphs[1,2].set_axis_off()    

    # Perform K-means clustering and plot the results
    movies_pred = kmeans_model.fit_predict(genre_dataset)
    graphs[0,0].scatter(genre_dataset.iloc[:,0], genre_dataset.iloc[:,1], c=movies_pred)
    graphs[0,0].set_title("K-Means")

    # Perform GMM clustering and plot the results
    movies_pred = gmm_model.fit_predict(genre_dataset)
    graphs[0,1].scatter(genre_dataset.iloc[:,0], genre_dataset.iloc[:,1], c=movies_pred)
    graphs[0,1].set_title("GMM") 

    # Perform Mean-Shift clustering and plot the results
    movies_pred = ms_model.fit_predict(genre_dataset)
    graphs[0,2].scatter(genre_dataset.iloc[:,0], genre_dataset.iloc[:,1], c=movies_pred)
    graphs[0,2].set_title("Mean-Shift")

    # Perform DBSCAN clustering and plot the results
    movies_pred = dbscan_model.fit_predict(genre_dataset)
    graphs[1,0].scatter(genre_dataset.iloc[:,0], genre_dataset.iloc[:,1], c=movies_pred)
    graphs[1,0].set_title("DBSCAN") 

    # Perform Hierarchical Agglomerative Clustering (HAC) and plot the results
    movies_pred = hac_model.fit_predict(genre_dataset)
    graphs[1,1].scatter(genre_dataset.iloc[:,0], genre_dataset.iloc[:,1], c=movies_pred)
    graphs[1,1].set_title("HAC")

    # Increase the size of the graphs and then save them to the visualiztions/ directory
    plt.gcf().set_size_inches((12.80,7.20), forward=False)
    plt.savefig("../visualizations/{}.png".format(genre), bbox_inches='tight', dpi=100)
    #plt.show()    

