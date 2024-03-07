import h5py
import pandas as pd
import numpy as np
import time
import csv
import os
import random
import copy
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv
import seaborn as sns
import umap



from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from scipy.spatial import procrustes
from sklearn.metrics import confusion_matrix
import matplotlib.gridspec as gridspec

from CytofDR.evaluation import EvaluationMetrics as EM
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform

#temp
from sklearn.manifold import Isomap


class CIMURA:

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.high_dim_trained_h5_path = None
        self.low_dim_trained_h5_path = None
        self.centroids_df = None

    def _calc_centroid_high_dim(self, data):
            min_sum_squared_distance = float('inf')
            centroid_index = -1
            
            for i, point in enumerate(data):
                sum_squared_distance = np.sum((data - point) ** 2)
                if sum_squared_distance < min_sum_squared_distance:
                    min_sum_squared_distance = sum_squared_distance
                    centroid_index = i
            
            return data[centroid_index]
    
    def _calculate_geodesic_distances(self, data, n_neighbors=None):
        # Find the nearest neighbors for each point
        not_connected = True
        if n_neighbors == None:
            n_neighbors = np.max([int(np.sqrt(len(data))), 1])
        else:
            n_neighbors = np.min([n_neighbors, len(data)])

        while not_connected:
          nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
          distances, indices = nbrs.kneighbors(data)

          # Create a sparse graph in which edge weights are the distances between neighbors
          n_samples = data.shape[0]
          rows, cols, values = [], [], []
          for i in range(n_samples):
              for j, idx in enumerate(indices[i]):
                  rows.append(i)
                  cols.append(idx)
                  values.append(distances[i, j])

          graph = csr_matrix((values, (rows, cols)), shape=(n_samples, n_samples))

          # Compute geodesic distances using Fast Marching Method
          geodesic_distances = shortest_path(graph, method='auto', directed=False)

          if np.isinf(geodesic_distances).any():
              #print(f"Inf values found in Gb matrix neighborhood Graph not connected,increasing n_neighbors from {n_neighbors} to {n_neighbors+2}")
              n_neighbors = n_neighbors + 2
          else:
            break
        return geodesic_distances.astype(np.float32)
    
    def _streaming_isomap(self, xs, Xb, Yb, Gb, k=None):
      if k == None:  
        k = int(np.max([np.sqrt(len(Xb)), 1]))

      mean_G_b = np.mean(Gb, axis=1)
      Gb_mean_all = np.mean(Gb)

      nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(Xb)
      Ys = np.zeros(Yb.shape[1])
      low_dim_inverse = np.linalg.inv(Yb.T @ Yb) @ Yb.T
      ones_n = np.ones(Gb.shape[0])
      Gs = np.zeros(Xb.shape[0])

      # Step 1: KNN and distances
      distances, indices = nbrs.kneighbors(xs.reshape(1, -1))
      distances = distances.flatten()
      indices = indices.flatten()

      # Step 2: Approximating Geodesic Distances
      g = np.full(Gb.shape[0], np.inf)

      # Update g for each point in Xb
      for j in range(Gb.shape[0]):  # Iterate over all training points
          for idx, dist in zip(indices, distances):
              g[j] = min(g[j], dist + Gb[idx, j])
      Gs[:] = g.flatten()

      # Step 3: Computing New Coordinates
      mean_g = np.mean(g)
      c = 0.5 * (mean_g * ones_n - g - Gb_mean_all * ones_n + mean_G_b)
      p = low_dim_inverse @ c

      Y_hat = np.vstack([Yb, p.T])
      y_s = p.T - np.mean(Y_hat, axis=0)

      Ys[:] = y_s.flatten()

      return Ys
    
    def _scale_reduction(self, data, desired_max_distance):
        # Initialize the maximum distance
        current_max_distance = 0.0
        
        # Iterate over pairs of points to find the maximum distance
        for i in range(len(data)):
            for j in range(i+1, len(data)):
                distance_ij = np.linalg.norm(data[i] - data[j])
                if distance_ij > current_max_distance:
                    current_max_distance = distance_ij
        
        # Scale the points by dividing by the current maximum distance and then multiply by the desired maximum distance
        scaled_points = (data / current_max_distance) * desired_max_distance

        return scaled_points

    def load_training(self, high_dim_trained_h5_path, low_dim_trained_h5_path, centroids_csv_path):
        self.high_dim_trained_h5_path = high_dim_trained_h5_path
        self.low_dim_trained_h5_path = low_dim_trained_h5_path

        centroids_df = pd.read_csv(centroids_csv_path)
        self.centroids_ids = centroids_df["cluster"]
        self.centroids_data = centroids_df.iloc[:, 1:].values
        

    def h5_to_array(self, h5_path):
        with h5py.File(h5_path, "r") as h5_file:
            entry_ids = []
            row_list = []
            for dataset_name in h5_file.keys():
                entry_ids.append(dataset_name)
                row = h5_file[dataset_name]
                row_list.append(np.array(row).astype(np.float32))
        data_2darray = np.vstack(row_list).astype(np.float32)
        return np.array(entry_ids), data_2darray


    def fit_transform(self, data, n_centroids=10, local_reducer=None, scaling_factor=0.1):
        #Region Approximation
        input_centroid = self._calc_centroid_high_dim(data)
        nbrs = NearestNeighbors(n_neighbors=n_centroids, algorithm='auto').fit(self.centroids_data)
        _,indices = nbrs.kneighbors([input_centroid])
        indices = indices.flatten()
        centroid_neighbors_ids = self.centroids_ids[indices]


        #Load Region specific Training Matrices
        #LOW-DIM
        data_arrays = []
        not_found = []
        with h5py.File(self.low_dim_trained_h5_path, 'r') as file:
            for cluster_id in centroid_neighbors_ids:
                cluster_id = f"cluster_{cluster_id}"
                if cluster_id in file:
                    data_arrays.append(file[cluster_id][:])
                else:
                    print(f"Low-Dim-WARNING: Dataset {cluster_id} not found in the file.")
                    not_found.append(cluster_id)

        training_low_dim = np.vstack(data_arrays)
        centroid_neighbors_ids = [item for item in centroid_neighbors_ids if item not in not_found]

        #HIGH-DIM
        data_arrays = []
        with h5py.File(self.high_dim_trained_h5_path, 'r') as file:
            for cluster_id in centroid_neighbors_ids:
                cluster_id = f"cluster_{cluster_id}"
                if cluster_id in file:
                    data_arrays.append(file[cluster_id][:])
                else:
                    print(f"High-Dim-WARNING: Dataset {cluster_id} not found in the file.")

        training_high_dim = np.vstack(data_arrays)

        #DIST
        training_geo_dist = self._calculate_geodesic_distances(training_high_dim)

        #Check shapes
        if training_high_dim.shape[0] == training_low_dim.shape[0] and training_geo_dist.shape[0] == training_low_dim.shape[0]:
            #print("Training Matrices Shapes are correct!")
            pass
        else:
            print("Training Matrices Shapes are NOT correct!")
            print(f"High_dim: {training_high_dim.shape[0]}")
            print(f"Low_dim: {training_low_dim.shape[0]}")
            print(f"Dist: {training_geo_dist.shape[0]}")


        #Reduce Input Dataset
        if local_reducer == None:
            local_reducer = umap.UMAP(n_neighbors=25, min_dist=0.5, n_components=self.n_components)
            #local_reducer = Isomap(n_components=self.n_components, n_neighbors=int(np.sqrt(len(data))))
        

        if len(data) <= local_reducer.n_neighbors:
            reduced_points = []
            for point in data:
                reduced_point = self._streaming_isomap(point.flatten(), training_high_dim, training_low_dim, training_geo_dist)
                reduced_points.append(reduced_point.reshape(1, -1))
            return np.vstack(reduced_points)
        

        #Project reduced_input_centroid using S-Isomap
        s_isomap_input_centroid = self._streaming_isomap(input_centroid, training_high_dim, training_low_dim, training_geo_dist)
        
        
        reduced_input = local_reducer.fit_transform(data)

        #Scale
        x_values = reduced_input[:, 0]
        y_values = reduced_input[:, 1]

        scaled_x_values = x_values * scaling_factor
        scaled_y_values = y_values * scaling_factor

        scaled_points_array = np.column_stack((scaled_x_values, scaled_y_values))

        if self.n_components == 3:
            z_values = reduced_input[:, 2]
            scaled_z_values = z_values * scaling_factor
            scaled_points_array = np.column_stack((scaled_x_values, scaled_y_values, scaled_z_values))

        else:
            scaled_points_array = np.column_stack((scaled_x_values, scaled_y_values))


        

        #Shift
        reduced_input_centroid = np.mean(scaled_points_array, axis=0)
        shift_vector = s_isomap_input_centroid - reduced_input_centroid
        shifted_reduced_input = scaled_points_array + shift_vector
        
        return shifted_reduced_input
    
    
    def fit_transform_csv_multi_dict(self, name_path_dict, output_csv_path, n_centroids=10, local_reducer=None, scaling_factor=0.1, mode="w"):
        if mode == "w":
            with open(output_csv_path, mode='w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow([f"dim{i}" for i in range(1, self.n_components+1)] + ["id", "group"])

        with open(output_csv_path, mode='a', newline='') as file:
            csv_writer = csv.writer(file)
            
            for name, path in name_path_dict.items():
                ids, data = self.h5_to_array(path)
                data_reduced = self.fit_transform(data, n_centroids=n_centroids, local_reducer=local_reducer, scaling_factor=scaling_factor)

                for reduced_data_row, id_value, name_value in zip(data_reduced, ids, [name]*len(data_reduced)):
                    csv_writer.writerow(list(reduced_data_row) + [id_value] + [name_value])


    def fit_transform_csv_multi_clusterfile(self, clusterfile, data_h5_path, output_csv_path, n_centroids=10, local_reducer=None, scaling_factor=0.1, mode="w"):
        if mode == "w":
            with open(output_csv_path, mode='w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow([f"dim{i}" for i in range(1, self.n_components+1)] + ["id", "group"])

        with open(output_csv_path, mode='a', newline='') as file:
            # Step 1: Read the CSV file
            csv_data = pd.read_csv(clusterfile)

            # Step 2: Open the HDF5 file
            with h5py.File(data_h5_path, 'r') as hf:
                # Group the DataFrame by cluster
                grouped_data = csv_data.groupby('cluster')

                # Iterate over groups
                for cluster_label, group_df in grouped_data:
                    #try:
                    stacked_arrays = []
                    data_point_ids = []
                    for _, row in group_df.iterrows():
                        data_point_id = row['id']

                        dataset_name = str(data_point_id)  # Assuming dataset names match the IDs
                        if dataset_name in hf:
                            data_array = np.array(hf[dataset_name])  # Retrieve the 1D array
                            stacked_arrays.append(data_array)
                            data_point_ids.append(data_point_id)
                        else:
                            # Handle the case where the dataset doesn't exist
                            print(f"Dataset '{dataset_name}' not found in HDF5 file.")

                    # Stack the 1D arrays into a 2D numpy array for the current cluster
                    stacked_array = np.vstack(stacked_arrays)
                    csv_writer = csv.writer(file)
                    
                    data_reduced = self.fit_transform(stacked_array, n_centroids=n_centroids, local_reducer=local_reducer, scaling_factor=scaling_factor)
                    
                    for reduced_data_row, id_value, name_value in zip(data_reduced, data_point_ids, [cluster_label]*len(data_reduced)):
                        csv_writer.writerow(list(reduced_data_row) + [id_value] + [name_value])
                    # except:
                    #     print(f"{cluster_label} failed.")    
                    print(f"cluster_label: {cluster_label}")


    def fit_transform_multi_clusterfile(self, clusterfile, data_h5_path, n_centroids=10, local_reducer=None, scaling_factor=0.1):
        all_reduced_data = []  # List to collect all reduced data rows
        data_point_ids = []  # List to collect data point ids
        #group_labels = []  # List to collect group labels

        # Step 1: Read the CSV file
        csv_data = pd.read_csv(clusterfile)

        # Step 2: Open the HDF5 file
        with h5py.File(data_h5_path, 'r') as hf:
            # Group the DataFrame by cluster
            grouped_data = csv_data.groupby('cluster')

            # Iterate over groups
            for cluster_label, group_df in grouped_data:
                stacked_arrays = []

                for _, row in group_df.iterrows():
                    data_point_id = row['id']

                    dataset_name = str(data_point_id)  # Assuming dataset names match the IDs
                    if dataset_name in hf:
                        data_array = np.array(hf[dataset_name])  # Retrieve the 1D array
                        stacked_arrays.append(data_array)
                    else:
                        # Handle the case where the dataset doesn't exist
                        print(f"Dataset '{dataset_name}' not found in HDF5 file.")

                # Stack the 1D arrays into a 2D numpy array for the current cluster
                if stacked_arrays:  # Ensure there is data to process
                    stacked_array = np.vstack(stacked_arrays)

                    data_reduced = self.fit_transform(stacked_array, n_centroids=n_centroids, local_reducer=local_reducer, scaling_factor=scaling_factor)

                    # Collect the reduced data, ids, and group labels for later stacking
                    all_reduced_data.extend(data_reduced)
                    data_point_ids.extend(group_df['id'].values)
                    #group_labels.extend([cluster_label] * len(data_reduced))

        # Stack the collected reduced data rows into a 2D numpy array
        reduced_data_array = np.array(all_reduced_data)
        ids_array = np.array(data_point_ids)

        # If you need to return the IDs and labels as well, you can adjust the return statement as needed
        return reduced_data_array, ids_array