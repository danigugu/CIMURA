import h5py
import pandas as pd
import numpy as np

from sklearn.cluster import MiniBatchKMeans

from sklearn.manifold import Isomap
from sklearn.decomposition import PCA

class CIMURA_Trainer:

    def __init__(self, n_components=2):
        self.high_dim_data = None
        self.high_dim_ids = None
        self.cluster_df = None
        self.centroids_df = None
        self.centroids_reduced_df = None

        self.trained_high_dim_h5_path = None
        self.trained_low_dim_h5_path = None
        self.trained_low_dim_shifted_h5_path = None

        self.n_components = n_components

    # "Private" Help Functions

    def _h5_to_array(self, h5_path):
        with h5py.File(h5_path, "r") as h5_file:
            entry_ids = []
            row_list = []
            for dataset_name in h5_file.keys():
                entry_ids.append(dataset_name)
                row = h5_file[dataset_name]
                row_list.append(np.array(row).astype(np.float32))
        data_2darray = np.vstack(row_list).astype(np.float32)
        return np.array(entry_ids), data_2darray
    

    def _calc_centroid_high_dim(self, data):
            min_sum_squared_distance = float('inf')
            centroid_index = -1
            
            for i, point in enumerate(data):
                sum_squared_distance = np.sum((data - point) ** 2)
                if sum_squared_distance < min_sum_squared_distance:
                    min_sum_squared_distance = sum_squared_distance
                    centroid_index = i
            
            return data[centroid_index]
        
    def _calc_centroid_low_dim(self, data):
        centroid = np.mean(data, axis=0)
        return centroid
    

    def _shift_cluster_centroids(self, low_dim_h5_file_path, centroid_red_df, shifted_h5_file_path):
        # Open HDF5 files
        with h5py.File(low_dim_h5_file_path, 'r') as low_dim_h5_file, \
                h5py.File(shifted_h5_file_path, 'w') as shifted_h5_file:
            for cluster_label in centroid_red_df['cluster'].unique():
                # Load low-dimensional data for the current cluster
                cluster_dataset_name = f'cluster_{cluster_label}'
                cluster_data = low_dim_h5_file[cluster_dataset_name][:]

                # Calculate centroid shift
                if self.n_components == 2:
                    cluster_centroid = centroid_red_df[centroid_red_df['cluster'] == cluster_label][['dim1', 'dim2']].values.flatten()
                elif self.n_components == 3:
                    cluster_centroid = centroid_red_df[centroid_red_df['cluster'] == cluster_label][['dim1', 'dim2', 'dim3']].values.flatten()

                data_centroid = self._calc_centroid_low_dim(cluster_data)
                centroid_shift = cluster_centroid - data_centroid

                # Shift the data points
                shifted_cluster_data = cluster_data + centroid_shift

                # Save the shifted data to the shifted HDF5 file
                shifted_h5_file.create_dataset(cluster_dataset_name, data=shifted_cluster_data)


    # "Public" Training Functions
    def load_training_data(self, high_dim_h5_path):
        self.high_dim_ids, self.high_dim_data = self._h5_to_array(high_dim_h5_path)


    def cluster_training_data(self, n_clusters, save_file_csv_path=None, desired_variance=0.95, batch_size=20000):
        # Fit PCA to estimate the best number of components
        pca = PCA().fit(self.high_dim_data)

        # Calculate the cumulative sum of explained variance ratio
        cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

        # Determine the number of components needed to explain desired variance, e.g., 95%
        components_for_desired_variance = np.where(cumulative_explained_variance >= desired_variance)[0][0] + 1

        print(f"Number of components to explain {desired_variance*100}% of variance: {components_for_desired_variance}")

        # Apply PCA with the estimated number of components
        pca_reduced = PCA(n_components=components_for_desired_variance)
        data_reduced = pca_reduced.fit_transform(self.high_dim_data)

        # Perform Mini-Batch KMeans clustering
        mini_batch_kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, compute_labels=True)
        cluster_labels = mini_batch_kmeans.fit_predict(data_reduced)

        # Create a DataFrame with cluster labels and row identifiers
        cluster_df = pd.DataFrame({'cluster': cluster_labels, 'id': self.high_dim_ids})

        if save_file_csv_path != None:
            cluster_df.to_csv(save_file_csv_path, index=False)
        
        self.cluster_df = cluster_df

    
    def calc_centroid_each_cluster(self, cluster_df, high_dim_data, save_file_csv_path=None):
        centroids = []
        for cluster_label in cluster_df['cluster'].unique():
            cluster_indices = cluster_df[cluster_df['cluster'] == cluster_label].index
            cluster_data = high_dim_data[cluster_indices]
            centroid = self._calc_centroid_high_dim(cluster_data)
            centroids.append({'cluster': cluster_label, 'centroid': centroid})

        centroids_df = pd.DataFrame(centroids)

        num_dimensions = high_dim_data.shape[1]
        centroids_df = pd.concat([centroids_df, pd.DataFrame(centroids_df['centroid'].tolist(), columns=[f'dim_{i+1}' for i in range(num_dimensions)])], axis=1)


        centroids_df.drop(columns=['centroid'], inplace=True)

        if save_file_csv_path != None:
            centroids_df.to_csv(save_file_csv_path, index=False)
        
        self.centroids_df = centroids_df


    def reduce_centroids(self, centroids_df, method="isomap", save_file_csv_path=None):
        centroids = centroids_df.drop(columns=['cluster']).values

        if method == "isomap":
            isomap = Isomap(n_components=self.n_components, n_neighbors=int(np.sqrt(centroids.shape[0]))) 
            centroids_reduced = isomap.fit_transform(centroids)
        if method == "pca":
            pca = PCA(n_components=self.n_components) 
            centroids_reduced = pca.fit_transform(centroids)

        centroids_reduced_df = pd.DataFrame(centroids_reduced, columns=[f"dim{i+1}" for i in range(self.n_components)])
        centroids_reduced_df.insert(0, 'cluster', centroids_df['cluster'])

        if save_file_csv_path != None:
            centroids_reduced_df.to_csv(save_file_csv_path, index=False)

        self.centroids_reduced_df = centroids_reduced_df

    def create_high_and_low_h5(self, cluster_df, high_dim_data, high_dim_trained_h5_file_path, low_dim_trained_h5_file_path, method="isomap", shift_low=True):
        # Save HDF5 paths
        self.trained_high_dim_h5_path = high_dim_trained_h5_file_path
        self.trained_low_dim_h5_path = low_dim_trained_h5_file_path
        self.trained_low_dim_shifted_h5_path = f"{low_dim_trained_h5_file_path.strip('.h5')}_shifted.h5"

        with h5py.File(low_dim_trained_h5_file_path, 'w') as low_dim_h5_file, \
                h5py.File(high_dim_trained_h5_file_path, 'w') as high_dim_h5_file:
            for cluster_label in cluster_df['cluster'].unique():
                # Get data points for the current cluster
                cluster_indices = cluster_df[cluster_df['cluster'] == cluster_label].index
                cluster_data = high_dim_data[cluster_indices]


                if len(cluster_data) <= 3:
                    cluster_data_low_dim = np.zeros((len(cluster_data), self.n_components))

                else:
                    if method == "isomap":
                        dim_red = Isomap(n_components=self.n_components, n_neighbors=np.max([int(np.sqrt(len(cluster_data))), 1]))
                    else:
                        print("Unvalid Method!")
                        break
                    cluster_data_low_dim = dim_red.fit_transform(cluster_data)

                # Save the Isomap coordinates to the Isomap HDF5 file
                cluster_dataset_name = f'cluster_{cluster_label}'
                low_dim_h5_file.create_dataset(cluster_dataset_name, data=cluster_data_low_dim)

                # Save the high-dimensional data to the high-dimensional HDF5 file
                high_dim_h5_file.create_dataset(cluster_dataset_name, data=cluster_data)
            
        if shift_low:
            self._shift_cluster_centroids(low_dim_trained_h5_file_path, self.centroids_reduced_df, f"{low_dim_trained_h5_file_path.strip('.h5')}_shifted.h5")

        
    def full_training(self, training_name, high_dim_training_h5_path, file_save_folder):
        self.load_training_data(high_dim_training_h5_path)
        self.cluster_training_data(10000, save_file_csv_path=f"{file_save_folder}/{training_name}_cluster.csv", desired_variance=0.95, batch_size=20000)
        self.calc_centroid_each_cluster(self.cluster_df, self.high_dim_data, save_file_csv_path=f"{file_save_folder}/{training_name}_centroids.csv")
        self.reduce_centroids(self.centroids_df, method="isomap", save_file_csv_path=f"{file_save_folder}/{training_name}_centroids_reduced.csv")
        self.create_high_and_low_h5(self.cluster_df, self.high_dim_data, f"{file_save_folder}/{training_name}_high_dim_trained.h5",
                                    f"{file_save_folder}/{training_name}_low_dim_trained.h5", method="isomap", shift_low=True)
        print("Training Completed!")