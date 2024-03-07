#IMPORTS
import pandas as pd
import numpy as np
import h5py
import datetime
import time
from CytofDR import dr
import os
from memory_profiler import profile
from CIMURA import CIMURA
import umap
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans


class Method_Runner():

    def __init__(self, data, ids, h5_path, methods_param_dict, out_dims):
        self.data = data
        self.ids = ids
        self.h5_path = h5_path
        self.methods_param_dict = methods_param_dict
        self.out_dims = out_dims
    
    def cluster_training_data(self, data, n_clusters, save_file_csv_path=None, desired_variance=0.95, batch_size=20000):
        # Fit PCA to estimate the best number of components
        pca = PCA().fit(data)

        # Calculate the cumulative sum of explained variance ratio
        cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

        # Determine the number of components needed to explain desired variance, e.g., 95%
        components_for_desired_variance = np.where(cumulative_explained_variance >= desired_variance)[0][0] + 1

        print(f"Number of components to explain {desired_variance*100}% of variance: {components_for_desired_variance}")

        # Apply PCA with the estimated number of components
        pca_reduced = PCA(n_components=components_for_desired_variance)
        data_reduced = pca_reduced.fit_transform(data)

        # Perform Mini-Batch KMeans clustering
        mini_batch_kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, compute_labels=True)
        cluster_labels = mini_batch_kmeans.fit_predict(data_reduced)

        # Create a DataFrame with cluster labels and row identifiers
        cluster_df = pd.DataFrame({'cluster': cluster_labels, 'id': self.ids})

        if save_file_csv_path != None:
            cluster_df.to_csv(save_file_csv_path, index=False)
        
    

    @profile
    def run_cimura(self, data, h5_path, methods_param_dict, out_dims, log_csv_path):
        cluster_path = "/mnt/project/guerguerian/swissprot_subsets/runtime_memory_logs/temp_cimura_clusters.csv"
        self.cluster_training_data(data, int(len(data)*0.001), save_file_csv_path=cluster_path)
        
        cimura = CIMURA(n_components=out_dims)
        high_dim_trained_h5_path = "/mnt/project/guerguerian/CIMURA/files/new_10k_isomap_prott5/new_10k_isomap_prott5_high_dim_trained.h5"
        low_dim_trained_h5_path = "/mnt/project/guerguerian/CIMURA/files/new_10k_isomap_prott5/new_10k_isomap_prott5_low_dim_trained_shifted.h5"
        centroids_csv_path = "/mnt/project/guerguerian/CIMURA/files/new_10k_isomap_prott5/new_10k_isomap_prott5_centroids.csv"
        cimura.load_training(high_dim_trained_h5_path, low_dim_trained_h5_path, centroids_csv_path)
        
        
        out_path = "/mnt/project/guerguerian/swissprot_subsets/runtime_memory_logs/temp_cimura_output.csv"

        local_reducer = umap.UMAP(n_neighbors=25, min_dist=0.5, n_components=2)
        st = time.process_time()
        cimura.fit_transform_csv_multi_clusterfile(clusterfile=cluster_path, data_h5_path=h5_path, output_csv_path=out_path, n_centroids=methods_param_dict["cimura"]["n_centroids"],
                                            local_reducer=local_reducer, scaling_factor=methods_param_dict["cimura"]["scaling_factor"])
        
        et = time.process_time()
        runtime = et - st

        log_df = pd.DataFrame({
            "method": ["cimura"],
            "data_size": [len(data)],
            "runtime": [runtime]
        }, index=[0])


        if os.path.exists(log_csv_path):
            log_df.to_csv(log_csv_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_csv_path, mode="a", header=True, index=False)

    @profile
    def run_umap(self, data, methods_param_dict, out_dims, log_csv_path):
        st = time.process_time()
        reduction = dr.NonLinearMethods.UMAP(data=data, out_dims=out_dims,
                                n_neighbors=methods_param_dict["umap"]["n_neighbors"],
                                min_dist=methods_param_dict["umap"]["min_dist"])
        et = time.process_time()
        runtime = et - st

        log_df = pd.DataFrame({
            "method": ["umap"],
            "data_size": [len(data)],
            "runtime": [runtime]
        }, index=[0])


        if os.path.exists(log_csv_path):
            log_df.to_csv(log_csv_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_csv_path, mode="a", header=True, index=False)            


    @profile
    def run_tsne_bh(self, data, methods_param_dict, out_dims, log_csv_path):
        st = time.process_time()
        reduction = dr.NonLinearMethods.sklearn_tsne(data=data, out_dims=out_dims,
                                                                perplexity=methods_param_dict["tsne_bh"]["perplexity"])
        et = time.process_time()
        runtime = et - st

        log_df = pd.DataFrame({
            "method": ["tsne_bh"],
            "data_size": [len(data)],
            "runtime": [runtime]
        }, index=[0])

        if os.path.exists(log_csv_path):
            log_df.to_csv(log_csv_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_csv_path, mode="a", header=True, index=False)

    @profile
    def run_tsne_fft(self, data, methods_param_dict, out_dims, log_csv_path):
        st = time.process_time()
        reduction = dr.NonLinearMethods.open_tsne(data=data, out_dims=out_dims,
                                                                perp=methods_param_dict["tsne_fft"]["perp"])
        et = time.process_time()
        runtime = et - st

        log_df = pd.DataFrame({
            "method": ["tsne_fft"],
            "data_size": [len(data)],
            "runtime": [runtime]
        }, index=[0])

        if os.path.exists(log_csv_path):
            log_df.to_csv(log_csv_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_csv_path, mode="a", header=True, index=False)

    @profile
    def run_phate(self, data, methods_param_dict, out_dims, log_csv_path):
        st = time.process_time()
        reduction = dr.NonLinearMethods.phate(data=data, out_dims=out_dims,
                                                    knn=methods_param_dict["phate"]["knn"],
                                                    gamma=methods_param_dict["phate"]["gamma"])
        et = time.process_time()
        runtime = et - st

        log_df = pd.DataFrame({
            "method": ["phate"],
            "data_size": [len(data)],
            "runtime": [runtime]
        }, index=[0])

        if os.path.exists(log_csv_path):
            log_df.to_csv(log_csv_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_csv_path, mode="a", header=True, index=False)

    @profile
    def run_isomap(self, data, methods_param_dict, out_dims, log_csv_path):
        print("HERE1")
        st = time.process_time()
        reduction = dr.NonLinearMethods.isomap(data=data, out_dims=out_dims,
                                                            n_neighbors=methods_param_dict["isomap"]["n_neighbors"])
        et = time.process_time()
        print("HERE2")
        runtime = et - st

        log_df = pd.DataFrame({
            "method": ["isomap"],
            "data_size": [len(data)],
            "runtime": [runtime]
        }, index=[0])

        if os.path.exists(log_csv_path):
            log_df.to_csv(log_csv_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_csv_path, mode="w", header=True, index=False)

    @profile
    def run_mds(self, data, methods_param_dict, out_dims, log_csv_path):
        st = time.process_time()
        reduction = dr.NonLinearMethods.MDS(data=data, out_dims=out_dims,
                                                        eps=methods_param_dict["mds"]["eps"])
        et = time.process_time()
        runtime = et - st

        log_df = pd.DataFrame({
            "method": ["mds"],
            "data_size": [len(data)],
            "runtime": [runtime]
        }, index=[0])

        if os.path.exists(log_csv_path):
            log_df.to_csv(log_csv_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_csv_path, mode="a", header=True, index=False)

    @profile
    def run_lle(self, data, methods_param_dict, out_dims, log_csv_path):
        st = time.process_time()
        reduction = dr.NonLinearMethods.LLE(data=data, out_dims=out_dims,
                                                    n_neighbors=methods_param_dict["lle"]["n_neighbors"],
                                                    reg=methods_param_dict["lle"]["reg"], eigen_solver="dense")
        et = time.process_time()
        runtime = et - st

        log_df = pd.DataFrame({
            "method": ["lle"],
            "data_size": [len(data)],
            "runtime": [runtime]
        }, index=[0])

        if os.path.exists(log_csv_path):
            log_df.to_csv(log_csv_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_csv_path, mode="a", header=True, index=False)

    @profile
    def run_kpca_poly(self, data, methods_param_dict, out_dims, log_csv_path):
        st = time.process_time()
        reduction = dr.NonLinearMethods.kernelPCA(data=data, out_dims=out_dims, kernel="poly",
                                                    gamma=methods_param_dict["kpca_poly"]["gamma"],
                                                    degree=methods_param_dict["kpca_poly"]["degree"])
        et = time.process_time()
        runtime = et - st

        log_df = pd.DataFrame({
            "method": ["kpca_poly"],
            "data_size": [len(data)],
            "runtime": [runtime]
        }, index=[0])

        if os.path.exists(log_csv_path):
            log_df.to_csv(log_csv_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_csv_path, mode="a", header=True, index=False)

    @profile
    def run_kpca_rbf(self, data, methods_param_dict, out_dims, log_csv_path):
        st = time.process_time()
        reduction = dr.NonLinearMethods.kernelPCA(data=data, out_dims=out_dims, kernel="rbf",
                                                    gamma=methods_param_dict["kpca_rbf"]["gamma"])
        et = time.process_time()
        runtime = et - st

        log_df = pd.DataFrame({
            "method": ["kpca_rbf"],
            "data_size": [len(data)],
            "runtime": [runtime]
        }, index=[0])

        if os.path.exists(log_csv_path):
            log_df.to_csv(log_csv_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_csv_path, mode="a", header=True, index=False)

    @profile
    def run_spectral(self, data, methods_param_dict, out_dims, log_csv_path):
        st = time.process_time()
        reduction = dr.NonLinearMethods.spectral(data=data, out_dims=out_dims,
                                                    n_neighbors=methods_param_dict["spectral"]["n_neighbors"],
                                                    gamma=methods_param_dict["spectral"]["gamma"])
        et = time.process_time()
        runtime = et - st

        log_df = pd.DataFrame({
            "method": ["spectral"],
            "data_size": [len(data)],
            "runtime": [runtime]
        }, index=[0])

        if os.path.exists(log_csv_path):
            log_df.to_csv(log_csv_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_csv_path, mode="a", header=True, index=False)

    @profile
    def run_pca(self, data, out_dims, log_csv_path):
        st = time.process_time()
        reduction = dr.LinearMethods.PCA(data=data, out_dims=out_dims)
        et = time.process_time()
        runtime = et - st

        log_df = pd.DataFrame({
            "method": ["pca"],
            "data_size": [len(data)],
            "runtime": [runtime]
        }, index=[0])

        if os.path.exists(log_csv_path):
            log_df.to_csv(log_csv_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_csv_path, mode="a", header=True, index=False)


    @profile
    def run_ica(self, data, out_dims, log_csv_path):
        st = time.process_time()
        reduction = dr.LinearMethods.ICA(data=data, out_dims=out_dims)
        et = time.process_time()
        runtime = et - st

        log_df = pd.DataFrame({
            "method": ["ica"],
            "data_size": [len(data)],
            "runtime": [runtime]
        }, index=[0])

        if os.path.exists(log_csv_path):
            log_df.to_csv(log_csv_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_csv_path, mode="a", header=True, index=False)


    @profile
    def run_fa(self, data, out_dims, log_csv_path):
        st = time.process_time()
        reduction = dr.LinearMethods.FA(data=data, out_dims=out_dims)
        et = time.process_time()
        runtime = et - st

        log_df = pd.DataFrame({
            "method": ["fa"],
            "data_size": [len(data)],
            "runtime": [runtime]
        }, index=[0])

        if os.path.exists(log_csv_path):
            log_df.to_csv(log_csv_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_csv_path, mode="a", header=True, index=False)


#self, data, methods_param_dict, out_dims, log_csv_path
    def run_methods(self, log_csv_path):
        data_len = len(self.data)
        try:
            print(f"CIMURA START {data_len}")
            self.run_cimura(self.data, self.h5_path, self.methods_param_dict, self.out_dims, log_csv_path)
            print("CIMURA DONE")
        except:
            print("CIMURA FAILED")

        try:
            print(f"UMAP START {data_len}")
            self.run_umap(self.data, self.methods_param_dict, self.out_dims, log_csv_path)
            print("UMAP DONE")
        except:
            print("UMAP FAILED")

        try:
            print(f"PCA START {data_len}")
            self.run_pca(self.data, self.out_dims, log_csv_path)
            print("PCA DONE")
        except:
            print("PCA FAILED")

        try:
            print(f"TSNE_BH START {data_len}")
            self.run_tsne_bh(self.data, self.methods_param_dict, self.out_dims, log_csv_path)
            print("TSNE_BH DONE")
        except:
            print("TSNE_BH FAILED")

        try:
           print(f"PHATE START {data_len}")
           self.run_phate(self.data, self.methods_param_dict, self.out_dims, log_csv_path)
           print("PHATE DONE")
        except:
           print("PHATE FAILED")
        
        try:
            print(f"ISOMAP START {data_len}")
            self.run_isomap(self.data, self.methods_param_dict, self.out_dims, log_csv_path)
            print("ISOMAP DONE")
        except:
            print("ISOMAP FAILED")

        # try:
        #     print("MDS START")
        #     self.run_mds(self.data, self.methods_param_dict, self.out_dims, log_csv_path)
        #     print("MDS DONE")
        # except:
        #     print("MDS FAILED")
        #self.run_tsne_fft(self.data, self.methods_param_dict, self.out_dims, log_csv_path)
        #self.run_spectral(self.data, self.methods_param_dict, self.out_dims, log_csv_path)
        #self.run_kpca_poly(self.data, self.methods_param_dict, self.out_dims, log_csv_path)
        #self.run_kpca_rbf(self.data, self.methods_param_dict, self.out_dims, log_csv_path)
        #self.run_lle(self.data, self.methods_param_dict, self.out_dims, log_csv_path)
        #self.run_ica(self.data, self.out_dims, log_csv_path)
        #self.run_fa(self.data, self.out_dims, log_csv_path)


if __name__ == "__main__":

    def h5_to_array(h5_path):
        with h5py.File(h5_path, "r") as h5_file:
            entry_ids = []
            row_list = []
            for dataset_name in h5_file.keys():
                entry_ids.append(dataset_name)
                row = h5_file[dataset_name]
                row_list.append(np.array(row).astype(np.float32))
        data_2darray = np.vstack(row_list).astype(np.float32)
        return np.array(entry_ids), data_2darray


    #h5_dict_local = {
    #"Pla2g2_esm2": r"C:\Users\dguer\Desktop\Bachelor_Thesis\Datasets\Pla2g2_esm2.h5",
    #"Pla2g2_esm2": r"C:\Users\dguer\Desktop\Bachelor_Thesis\Datasets\Pla2g2_esm2.h5",
    #"KLK_esm2": r"C:\Users\dguer\Desktop\Bachelor_Thesis\Datasets\KLK_esm2.h5",
    #"KLK_esm2": r"C:\Users\dguer\Desktop\Bachelor_Thesis\Datasets\KLK_esm2.h5",
    #"3FTx_mature_esm2": r"C:\Users\dguer\Desktop\Bachelor_Thesis\Datasets\3FTx_mature_esm2.h5",
    #"3FTx_mature_esm2": r"C:\Users\dguer\Desktop\Bachelor_Thesis\Datasets\3FTx_mature_esm2.h5"
    #}

    # h5_dict_cluster = {
    # #"Pla2g2_esm2": r"/mnt/project/guerguerian/benchmark/datasets/Pla2g2_esm2.h5",
    # "Pla2g2_esm2": r"/mnt/project/guerguerian/benchmark/datasets/Pla2g2_esm2.h5",
    # #"KLK_esm2": r"/mnt/project/guerguerian/benchmark/datasets/KLK_esm2.h5",
    # "KLK_esm2": r"/mnt/project/guerguerian/benchmark/datasets/KLK_esm2.h5",
    # #"3FTx_mature_esm2": r"/mnt/project/guerguerian/benchmark/datasets/3FTx_mature_esm2.h5",
    # "3FTx_mature_esm2": r"/mnt/project/guerguerian/benchmark/datasets/3FTx_mature_esm2.h5"
    # }

    h5_dict_cluster = {
    # "swissprot_subset_10000": r"/mnt/project/guerguerian/swissprot_subsets/subsets/prott5_subsets/swissprot_subset_prott5_10000.h5",
    # "swissprot_subset_20000": r"/mnt/project/guerguerian/swissprot_subsets/subsets/prott5_medium_subsets/swissprot_subset_prott5_20000.h5",
    # "swissprot_subset_40000": r"/mnt/project/guerguerian/swissprot_subsets/subsets/prott5_medium_subsets/swissprot_subset_prott5_40000.h5",
    # "swissprot_subset_60000": r"/mnt/project/guerguerian/swissprot_subsets/subsets/prott5_medium_subsets/swissprot_subset_prott5_60000.h5",
    # "swissprot_subset_80000": r"/mnt/project/guerguerian/swissprot_subsets/subsets/prott5_medium_subsets/swissprot_subset_prott5_80000.h5",
    # #"swissprot_subset_100000_n50": r"/mnt/project/guerguerian/swissprot_subsets/subsets/prott5_medium_subsets/swissprot_subset_prott5_100000.h5",
    # "swissprot_subset_100000": r"/mnt/project/guerguerian/swissprot_subsets/subsets/prott5_medium_subsets/swissprot_subset_prott5_100000.h5"#,
    #"swissprot_full": r"/mnt/project/senoner/datasets/swiss_prot/20230913_uniprot_sprot_esm2_3b.h5"
    "swissprot_subset_200000": "/mnt/project/guerguerian/swissprot_subsets/subsets/prott5_large_subsets/swissprot_subset_prott5_200000.h5",
    "swissprot_subset_300000": "/mnt/project/guerguerian/swissprot_subsets/subsets/prott5_large_subsets/swissprot_subset_prott5_300000.h5",
    "swissprot_subset_400000": "/mnt/project/guerguerian/swissprot_subsets/subsets/prott5_large_subsets/swissprot_subset_prott5_400000.h5",
    "swissprot_subset_500000": "/mnt/project/guerguerian/swissprot_subsets/subsets/prott5_large_subsets/swissprot_subset_prott5_500000.h5"
    }

    out_dims = 2
    #log_csv_path = r"C:\Users\dguer\Desktop\Bachelor_Thesis\cluster_configs\swissprot_subsets\test_log.csv"
    log_csv_path = r"/mnt/project/guerguerian/swissprot_subsets/runtime_memory_logs/all_new_runtime_logs_prott5_gpu_big.csv"

    for ds, path in h5_dict_cluster.items():
        print(f"{ds} start")
        try:
            ids, data = h5_to_array(path)
        except:
            print(f"{ds} CANT BE LOADED")
            continue

        n_neighbors = 50
        
        
        
        methods_param_dict = {
            #Tuneable methods
            "cimura": {
                "n_centroids": 5,
                "scaling_factor": 0.02
            },
            "umap": {
                "n_neighbors": n_neighbors,
                "min_dist":  0.1
            },
            "tsne_bh":{
                "perplexity": float(n_neighbors)
            },
            "phate":{
                "knn": n_neighbors,
                "gamma": 1
            },
            "isomap":{
                "n_neighbors": n_neighbors
            },
            "mds":{
                "eps": 0.0001
            },
            "lle":{
                "n_neighbors": 28,
                "reg": 0.06
            },
            "kpca_poly": {
                "gamma": 2.77,
                "degree": 3
            },
            "kpca_rbf": {
                "gamma": 2.42
            },
            "spectral": {
                "n_neighbors": 12,
                "gamma": 6.12
            },
            #Additional methods
            "pca": {},
            "ica": {},
            "fa": {}
        }
        
        runner = Method_Runner(data=data, ids=ids, h5_path=path, methods_param_dict=methods_param_dict,
                    out_dims=out_dims)
        

        runner.run_methods(log_csv_path)
        
        print(f"{ds} done")

