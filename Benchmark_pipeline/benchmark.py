#IMPORTS
import pandas as pd
import numpy as np
import h5py
import math
import matplotlib.pyplot as plt
import datetime
import time
import seaborn as sns
import os
from CytofDR import dr
from scipy.stats import spearmanr
from CytofDR.evaluation import EvaluationMetrics as EM
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
import umap

from CIMURA import CIMURA


#DataHandler
class DimRedDataHandler:
    def __init__(self, h5_path, n_groups, out_dims, dataset_name, id_path, clusterfile=None):
        self.ids, self.data = self.h5_to_2darray(h5_path)
        self.h5_path = h5_path
        self.id_path = id_path
        self.n_groups=n_groups
        self.out_dims=out_dims
        self.dataset_name = dataset_name
        self.high_dim_clusters = self.cluster_custom(self.data, "kmeans")
        self.distance_matrix_high_dim = squareform(pdist(self.data, metric='euclidean'))
        self.distance_vector_high_dim = self.distance_matrix_high_dim[np.triu_indices(self.distance_matrix_high_dim.shape[0], k=1)]
        self.original_neighbors = EM.build_annoy(self.data, k=math.ceil(len(self.data) * 0.01))
        self.ranked_distance_vector_high_dim = self.distance_vector_high_dim.argsort().argsort()
        self.save_ids_and_clusters(self.ids, self.high_dim_clusters, id_path, "high_dim")
        self.clusterfile = clusterfile

    def h5_to_2darray(self, h5_path):
        with h5py.File(h5_path, "r") as h5_file:
            entry_ids = []
            row_list = []
            for dataset_name in h5_file.keys():
                entry_ids.append(dataset_name)
                row = h5_file[dataset_name]
                row_list.append(np.array(row))
        data_2darray = np.vstack(row_list)
        return entry_ids, data_2darray
    
    def save_ids_and_clusters(self, id_list, cluster_list, id_path, dim, emb_name=""):
        if emb_name == "":
            id_path = id_path + f"/{dim}/{self.dataset_name}_ids.txt"
        else:
            id_path = id_path + f"/{dim}/{emb_name}_{self.dataset_name}_ids.txt"
        with open(id_path, "w") as file:
            for i in range(len(id_list)):
                file.write(id_list[i] + "," + str(cluster_list[i]) + "\n")


    def cluster_custom(self, data, method):
        if method == "dbscan":
            dbscan = DBSCAN()
            clusters = DBSCAN.fit_predict(data)
        elif method == "kmeans":
            kmeans = KMeans(n_clusters=self.n_groups)
            clusters = kmeans.fit_predict(data)
        return clusters
        
    
    def calculate_distance_matrix(self, array):
            #Calculate the pairwise Euclidean distance using broadcasting
            diff = array[:, np.newaxis, :] - array[np.newaxis, :, :]
            distance_matrix = np.sqrt(np.sum(diff**2, axis=-1))
            return distance_matrix
    
#Benchmarker
class DimRedBenchmarker:
    
    def __init__(self):
        pass

    @staticmethod
    def get_smaller_better_metrics():
        return ["emd", "npe", "cluster reconstruction: DBI", "emd_p2p"]
    

    @staticmethod
    def get_rank_dict(reduction):
        return reduction.rank_dr_methods()
    
    @staticmethod
    def get_best_method(reduction):
        rank_dict = reduction.rank_dr_methods()
        return max(rank_dict, key=rank_dict.get)
    

    @staticmethod
    def get_dataset_palette():
        palette_dict = {
            #Blue
            "Pla2g2_prott5": "#220DE4",
            "Pla2g2_esm2": "#2F8BB3",
            #Green
            "KLK_prott5": "#1FBA53",
            "KLK_esm2": "#7BF0A2",
            #Pink
            "3FTx_mature_prott5": "#EC43D6",
            "3FTx_mature_esm2": "#F9A9EE",
        }

        return palette_dict
        
    
    @staticmethod
    def get_eval_df(reduction):
        evals = reduction.evaluations
        rank_dict = reduction.rank_dr_methods()

        if len(reduction.custom_evaluations["custom"]) != 0:
            evals["custom"] = reduction.custom_evaluations["custom"]
            custom_rank_dict = reduction.rank_dr_methods_custom()
        
        data = {}

        example_category = next(iter(evals.keys()))
        example_metric = next(iter(evals[example_category].keys()))

        methods = evals[example_category][example_metric].keys()
            
        for method in methods:
            metric_values = {}
            for metric_type in evals:
                for metric in evals[metric_type]:
                    if len(evals[metric_type][metric]) != 0:
                        metric_values[metric] = evals[metric_type][metric][method]
            data[method] = metric_values

        df = pd.DataFrame(data)
        df = df.transpose()
        df["rank"] = df.index.map(rank_dict)

        if len(reduction.custom_evaluations["custom"]) != 0:
            df["rank_custom"] = df.index.map(custom_rank_dict)

        return df
    
    @staticmethod
    def eval_df_to_rank_metric_df(df):
        df = df.sort_index()
        def custom_rank(col):
            if col.name in DimRedBenchmarker.get_smaller_better_metrics():
                return col.rank(ascending=False)
            else:
                return col.rank()
    
        return df.apply(custom_rank)
    
    
    def get_eval_avg_df(df_list):
        if not DimRedBenchmarker.are_columns_in_same_order(df_list):
            print("DataFrame Columns are not in same order!")
            return None
      
        average_df = pd.concat(df_list).groupby(level=0).mean()
        if "rank_custom" in average_df.columns:
                average_df = average_df.sort_values(by="rank_custom", ascending=False)
                average_df = average_df.drop("rank_custom", axis=1)
        elif "rank" in average_df.columns:
                average_df = average_df.sort_values(by="rank", ascending=False)
                average_df = average_df.drop("rank", axis=1)
        return average_df
    

    def get_eval_multi2single_df(df_list, dataset_list):
        if not DimRedBenchmarker.are_columns_in_same_order(df_list):
            print("DataFrame Columns are not in same order!")
            return None
        
        n_rows = len(df_list[0])
        index = list(df_list[0].index)
        ds_col = [ds for ds in dataset_list for _ in range(n_rows)]
        met_col = index * len(dataset_list)

        result_df = pd.concat(df_list, ignore_index=True)
        result_df["dataset"] = ds_col
        result_df["method"] = met_col
        
        return result_df
    


    def get_metric_class_df(eval_df):
        metric_classes = {
            "spearman": "global",
            "emd": "global",
            "knn": "local",
            "npe": "local",
            "cluster reconstruction: silhouette": "downstream",
            "cluster reconstruction: DBI": "downstream",
            "cluster reconstruction: CHI": "downstream",
            "cluster reconstruction: RF": "downstream",
            "cluster concordance: ARI": "downstream",
            "cluster concordance: NMI": "downstream",
            "rank": "rank",
            "spearman_p2p": "custom",
            "emd_p2p": "custom",
            "knn_0.01": "custom",
            "ari": "custom",
            "nmi": "custom",
            "rank_custom": "custom"
        }
        grouped = eval_df.groupby(metric_classes, axis=1).mean()
        return grouped


    @staticmethod
    def plot_reduction_accuracy(evals_df, show, save, metrics, dataset_name, single_param_dict, out_dims, save_folder=""):
        eval_df = evals_df.copy()
        metric_columns = metrics if metrics != "all" else eval_df.columns
        
        color = DimRedBenchmarker.get_dataset_color(dataset_name=dataset_name)

        #Add Parameters to index names
        method_list = []
        text_box_str = "Hyperparameters: \n"
        for method in eval_df.index:
            param_string = ""
            text_box_str += f"{method} -> "
            if method == "tsne_fft":
                tsne_list = single_param_dict["tsne_fft"]["perp"]
                param_string = str(min(tsne_list)) + "_" + str(max(tsne_list)) + "_" + str(len(tsne_list))

                text_box_str += f"perp: min={min(tsne_list)}, max={max(tsne_list)}, n_values={str(len(tsne_list))}"
            else:
                for metric in single_param_dict[method]:
                    param_string += "_" if (len(param_string) != 0) else ""
                    param_string += str(single_param_dict[method][metric])

                    text_box_str += f"{metric}: {single_param_dict[method][metric]}  "
            method_list.append(method + "\n" + param_string)
            text_box_str += "\n"

        eval_df.index = method_list
        if len(metric_columns) == 1:
            asc = True if metric_columns[0] in DimRedBenchmarker.get_smaller_better_metrics() else False
            sorted_data = eval_df.sort_values(by=metric_columns[0], ascending=asc)
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.bar(sorted_data.index, sorted_data[metric_columns[0]], color = color)
            plt.xlabel("method")
            plt.ylabel(metric_columns[0])
            plt.title(f"{dataset_name}: {metric_columns[0]}   out_dims={out_dims}")
            plt.xticks(rotation=60)
            plt.axhline(0, color="black", linestyle="--", linewidth=1)

        else:
            n_rows = int(math.ceil(len(metric_columns)/2))
            fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))
            for i, metric in enumerate(metric_columns):
                row_idx = i // 2
                col_idx = i % 2
                ax = axes[row_idx, col_idx]
                asc = True if metric in DimRedBenchmarker.get_smaller_better_metrics() else False
                sorted_data = eval_df.sort_values(by=metric, ascending=asc)
                ax.bar(sorted_data.index, sorted_data[metric], color = color)
                ax.set_title(f"{dataset_name}: {metric}   out_dims={out_dims}")
                ax.set_xlabel("method")
                ax.set_ylabel(metric)
                x_entries = sorted_data.index if len(metric_columns) % 2 == 0 else [x.split("\n")[0] for x in sorted_data.index]
                ax.set_xticklabels(x_entries, rotation=60)
                ax.axhline(0, color="black", linestyle="--", linewidth=1)
            
            if len(metric_columns) % 2 != 0:
                last_row = (len(metric_columns) - 1) // 2
                last_col = 1
                fig.delaxes(axes[last_row, last_col])

                fig.text(
                    ((last_col + 1) / 2) - 0.4,  # X-coordinate (position in the empty column)
                    (1 - (last_row + 1) / n_rows) + 0.1,  # Y-coordinate (position in the empty row)
                    text_box_str,  # Text to display
                    ha="left",  # Horizontal alignment
                    va="top",  # Vertical alignment
                    fontsize=14,
                    bbox=dict(boxstyle="round", facecolor="lightgray", edgecolor="gray")
                )

            plt.tight_layout()

        if save and save_folder != "":
            current_datetime = datetime.datetime.now()
            today_date = current_datetime.date().strftime("%Y-%m-%d")
            current_time = current_datetime.time().strftime("%H-%M-%S")
            plt.savefig(f"{save_folder}/plot_reduction/{today_date}_{current_time}_{dataset_name}_bar.png")

            print("reduction barplot saved.")


        ##multibar plot (bigger better and smaller better)
        bb_metrics = ["spearman_p2p","spearman"]
        sb_metrics = ["emd_p2p", "emd"]

        eval_df_sorted = eval_df.sort_values(by=["rank_custom"], ascending=False)
        
        fig, axs = plt.subplots(4, 1, figsize=(14, 14))

        # Plot the sim metrics in the first subplot
        eval_df_sorted[bb_metrics].plot(kind="bar", ax=axs[0], color=sns.color_palette("husl", 8))
        axs[0].set_title("Similarity Metrics")
        axs[0].legend(loc="upper left", bbox_to_anchor=(1, 1))
        axs[0].axhline(y=0.0, color="black", linestyle="--", label="Threshold Line")

        # Plot the dist metrics in the second subplot
        eval_df_sorted[sb_metrics].plot(kind="bar", ax=axs[1])
        axs[1].set_title("Distance Metrics")
        axs[1].legend(loc="upper left", bbox_to_anchor=(1, 1))

        # Plot the rank in the third subplot
        eval_df_sorted[["rank"]].plot(kind="bar", ax=axs[2])
        axs[2].set_title("Rank")
        axs[2].legend(loc="upper left", bbox_to_anchor=(1, 1))

        # Plot the rank_custom in the fourth subplot
        eval_df_sorted[["rank_custom"]].plot(kind="bar", ax=axs[3])
        axs[3].set_title("Rank_custom")
        axs[3].legend(loc="upper left", bbox_to_anchor=(1, 1))

        x_entries = [x.split("\n")[0] for x in eval_df_sorted.index]

        axs[0].set_xticklabels(x_entries, rotation=45, ha="right")
        axs[1].set_xticklabels(x_entries, rotation=45, ha="right")
        axs[2].set_xticklabels(x_entries, rotation=45, ha="right")
        axs[3].set_xticklabels(x_entries, rotation=45, ha="right")

        annotation_str = f"{text_box_str}\ndataset: {dataset_name}\nout_dims: {out_dims}"

        


        # Adjust the layout and labels as needed
        plt.suptitle(f"Single Parameter Set Reduction on {dataset_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{save_folder}/plot_reduction/category_multibar/{today_date}_{current_time}_{dataset_name}.png") 
        with open(f"{save_folder}/plot_reduction/category_multibar/{today_date}_{current_time}_annotation.txt", "w") as file:
            file.write(annotation_str)
        print("param tune category multibar barplot saved.")

        if show:
            #plt.show()
            pass


    @staticmethod
    def plot_param_tune_accuracy(eval_df, metrics, dataset_name, out_dims, save_folder, n_param_sets = 5, method=""):
        metric_columns = metrics if metrics != "all" else eval_df.columns
        color = DimRedBenchmarker.get_dataset_color(dataset_name=dataset_name)
        eval_df = eval_df.sort_values(by="rank_custom", ascending=False) if "rank_custom" in metric_columns else eval_df.sort_values(by="rank", ascending=False)
        eval_df = eval_df[:n_param_sets]

        
        if len(metric_columns) == 1:
            asc = True if metric_columns[0] in DimRedBenchmarker.get_smaller_better_metrics() else False
            sorted_data = eval_df.sort_values(by=metric_columns[0], ascending=asc)
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.bar(sorted_data.index, sorted_data[metric_columns[0]], color = color)
            plt.xlabel("method")
            plt.ylabel(metric_columns[0])
            plt.title(f"{dataset_name}: {metric_columns[0]}   out_dims={out_dims}")
            plt.xticks(rotation=60)
            plt.axhline(0, color="black", linestyle="--", linewidth=1)

        else:
            n_rows = int(math.ceil(len(metric_columns)/2))
            fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))
            for i, metric in enumerate(metric_columns):
                row_idx = i // 2
                col_idx = i % 2
                ax = axes[row_idx, col_idx]
                asc = True if metric in DimRedBenchmarker.get_smaller_better_metrics() else False
                sorted_data = eval_df.sort_values(by=metric, ascending=asc)
                ax.bar(sorted_data.index, sorted_data[metric], color = color)
                ax.set_title(f"{dataset_name}: {method}  {metric}   out_dims={out_dims}")
                ax.set_xlabel("parameters")
                ax.set_ylabel(metric)
                ax.set_xticklabels(sorted_data.index, rotation=60)
                ax.axhline(0, color="black", linestyle="--", linewidth=1)
            
            if len(metric_columns) % 2 != 0:
                last_row = (len(metric_columns) - 1) // 2
                last_col = 1
                fig.delaxes(axes[last_row, last_col])
                

                fig.text(
                    ((last_col + 1) / 2) - 0.4,  # X-coordinate (position in the empty column)
                    (1 - (last_row + 1) / n_rows) + 0.055,  # Y-coordinate (position in the empty row)
                    DimRedBenchmarker.get_method_parameter_str(method),  # Text to display
                    ha="left",  # Horizontal alignment
                    va="top",  # Vertical alignment
                    fontsize=14,
                    bbox=dict(boxstyle="round", facecolor="lightgray", edgecolor="gray")
                )

            plt.tight_layout()

        
        current_datetime = datetime.datetime.now()
        today_date = current_datetime.date().strftime("%Y-%m-%d")
        current_time = current_datetime.time().strftime("%H-%M-%S")
        plt.savefig(f"{save_folder}/plot_param_tune/{today_date}_{current_time}_{dataset_name}.png")
        print(f"param tune {method} barplot saved.")

        plt.close()
        #multibar plot (bigger better and smaller better)
        bb_metrics = ["spearman_p2p","spearman"]
        sb_metrics = ["emd_p2p", "emd"]

        eval_df_sorted = eval_df.sort_values(by="rank_custom", ascending=False) if "rank_custom" in metric_columns else eval_df.sort_values(by="rank", ascending=False)
        fig, axs = plt.subplots(4, 1, figsize=(10, 12))

        # Plot the sim metrics in the first subplot
        eval_df_sorted[bb_metrics].plot(kind="bar", ax=axs[0], color=sns.color_palette("husl", 8))
        axs[0].set_title("Similarity Metrics")
        axs[0].legend(loc="upper left", bbox_to_anchor=(1, 1))
        axs[0].axhline(y=0.0, color="black", linestyle="--", label="Threshold Line")

        # Plot the dist metrics in the second subplot
        eval_df_sorted[sb_metrics].plot(kind="bar", ax=axs[1])
        axs[1].set_title("Distance Metrics")
        axs[1].legend(loc="upper left", bbox_to_anchor=(1, 1))

        # Plot the rank in the third subplot
        eval_df_sorted[["rank"]].plot(kind="bar", ax=axs[2])
        axs[2].set_title("Rank")
        axs[2].legend(loc="upper left", bbox_to_anchor=(1, 1))

        # Plot the rank in the fourth subplot
        eval_df_sorted[["rank_custom"]].plot(kind="bar", ax=axs[3])
        axs[3].set_title("Rank_custom")
        axs[3].legend(loc="upper left", bbox_to_anchor=(1, 1))


        axs[0].set_xticklabels(eval_df_sorted.index, rotation=45, ha="right")
        axs[1].set_xticklabels(eval_df_sorted.index, rotation=45, ha="right")
        axs[2].set_xticklabels(eval_df_sorted.index, rotation=45, ha="right")
        axs[3].set_xticklabels(eval_df_sorted.index, rotation=45, ha="right")

        textbox_str = f"{DimRedBenchmarker.get_method_parameter_str(method)}\ndataset: {dataset_name}\nout_dims: {out_dims}"

        textbox = plt.text(1.1, -1.5, textbox_str,
                        transform=axs[1].transAxes, fontsize=12,
                        bbox=dict(boxstyle="round", facecolor="lightgray", edgecolor="gray"))


        # Adjust the layout and labels as needed
        plt.suptitle(f"{method} Hyperparameter Tuning old vs new", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{save_folder}/plot_param_tune/category_multibar/{today_date}_{current_time}_{dataset_name}_{method}.png") 
        print("param tune category multibar barplot saved.")


    @staticmethod
    def plot_reduction_accuracy_multi_avgrank_heatmap(avg_rank_df, save_folder, out_dims, annotation_str):
        plt.close()
        avg_rank_df = avg_rank_df[["spearman_p2p", "emd_p2p", "knn_0.01", "ari", "nmi", "cluster reconstruction: silhouette"]]
        data = avg_rank_df.values
        row_labels = avg_rank_df.index
        col_labels = [met.split(":")[-1] for met in avg_rank_df.columns]

        #PLOT
        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = sns.heatmap(data, annot=False, cmap="YlGnBu", square=True, xticklabels=col_labels, yticklabels=row_labels)

        cbar = heatmap.collections[0].colorbar
        cbar.set_label("Average Rank")
        plt.title("Average Rank over multiple Datasets")
        #####

        current_datetime = datetime.datetime.now()
        today_date = current_datetime.date().strftime("%Y-%m-%d")
        current_time = current_datetime.time().strftime("%H-%M-%S")
        plt.savefig(f"{save_folder}/plot_multi_avgrank_heatmap/{today_date}_{current_time}_heatmap.png")
        with open(f"{save_folder}/plot_multi_avgrank_heatmap/{today_date}_{current_time}_heatmap.txt", "w") as file:
            file.write(annotation_str)
            file.write(f"out_dims: {out_dims}")
            

    @staticmethod
    def plot_reduction_accuracy_multi_datasetbars(ds_eval_df_dict, save_folder, out_dims, annotation_str):
        plt.close()
        
        concatenated_df = pd.concat([df.assign(dataset=ds) for ds, df in ds_eval_df_dict.items()], ignore_index=False)
        concatenated_df["method"] = concatenated_df.index
        concatenated_df = concatenated_df[["method", "dataset", "spearman_p2p", "emd_p2p", "knn_0.01",
                                            "ari", "nmi", "cluster reconstruction: silhouette", "rank_custom"]]

        # PLOT
        dataset_palette = DimRedBenchmarker.get_dataset_palette()
        method_averages = concatenated_df.groupby("method")["rank_custom"].mean().reset_index()

        # Sort the methods by the average of rank in descending order
        sorted_methods = method_averages.sort_values(by="rank_custom", ascending=False)["method"]

        fig, axes = plt.subplots(7, 1, figsize=(14, 20))

        y_cols = ["spearman_p2p", "emd_p2p", "knn_0.01",
                "ari", "nmi", "cluster reconstruction: silhouette", "rank_custom"]

        for i, ax in enumerate(axes.flatten()):
            column = y_cols[i]

            # Create a barplot in the current subplot
            bar = sns.barplot(data=concatenated_df,x="method", y=column, hue="dataset", ax=ax, order=sorted_methods, palette=dataset_palette)
            ax.set_title(f"{y_cols[i]} Accuracy Scores per Dataset") if column != "rank" else ax.set_title(f"total {y_cols[i]} average")
            ax.set_xticklabels(bar.get_xticklabels(), rotation=60)
            ax.set_xlabel("")
        # Adjust the layout
        plt.tight_layout()

        # Create a single legend for all subplots, position it outside the axes
        handles, labels = ax.get_legend_handles_labels()
        legend = plt.figlegend(handles, labels, loc="upper right", bbox_to_anchor=(1.25, 0.95))
        for ax in axes.flat:
            ax.get_legend().remove()
        for text in legend.get_texts():
            text.set_fontsize(15)
        plt.text(1.1, 0.75, f"out_dims: {out_dims}", fontsize=15, verticalalignment="top", transform=fig.transFigure)
        plt.suptitle(f"Single Parameter Set Reduction over multiple Datasets", fontsize=16, y=1.02)

        plt.tight_layout()
       
        current_datetime = datetime.datetime.now()
        today_date = current_datetime.date().strftime("%Y-%m-%d")
        current_time = current_datetime.time().strftime("%H-%M-%S")
        plt.savefig(f"{save_folder}/plot_multi_datasetbars/{today_date}_{current_time}_multibar.png", bbox_inches="tight")
        with open(f"{save_folder}/plot_multi_datasetbars/{today_date}_{current_time}_multibar.txt", "w") as file:
            file.write(annotation_str)
            file.write(f"out_dims: {out_dims}")


    @staticmethod
    def get_dataset_color(dataset_name):
        palette_dict = DimRedBenchmarker.get_dataset_palette()
        if dataset_name in palette_dict:
            return palette_dict[dataset_name]
        else:
            return "#C17A1D"
        
    
    @staticmethod
    def get_method_parameter_str(method):
        if method == "umap":
            return "umap Hyperparameters: n_neighbors  _  min_dist"
        if method == "tsne_bh":
            return "tsne_bh Hyperparameters: perplexity"
        if method == "tsne_fft":
            return "tsne_fft Hyperparameters: perp"
        if method == "phate":
            return "phate Hyperparameters: knn  _  gamma"
        if method == "pca" or method == "ica" or method == "fa":
            return ""
        if method == "isomap":
            return "isomap Hyperparameters: n_neighbors"
        if method == "mds":
            return "mds Hyperparameters: eps"
        if method == "lle":
            return "lle Hyperparameters: n_neighbors  _  reg"
        if method == "kpca_poly":
            return "kpca_poly Hyperparameters: gamma  _  degree"
        if method == "kpca_rbf":
            return "kpca_rbf Hyperparameters: gamma"
        if method == "spectral":
            return "spectral Hyperparameters: n_neighbors  _  gamma"




    @staticmethod
    def are_columns_in_same_order(dataframes):
        if not dataframes:
            return True  # If the list is empty, all columns are in the same order (by definition)

        reference_columns = list(dataframes[0].columns)

        for df in dataframes[1:]:
            if list(df.columns) != reference_columns:
                return False

        return True
    
#Embedder
class DimRedEmbedder:
    
    def __init__(self, data_handler, log_folder_path, plot_folder_path):
       self.data_handler = data_handler
       self.log_folder_path = log_folder_path
       self.plot_folder_path = plot_folder_path

    def add_custom_evals(self, results):
        def calculate_distance_matrix(array):
            #Calculate the pairwise Euclidean distance using broadcasting
            diff = array[:, np.newaxis, :] - array[np.newaxis, :, :]
            distance_matrix = np.sqrt(np.sum(diff**2, axis=-1))
            return distance_matrix
        
        distance_vector_high_dim = self.data_handler.distance_vector_high_dim
        original_neighbors = self.data_handler.original_neighbors
        ranked_distance_vector_high_dim = self.data_handler.ranked_distance_vector_high_dim
        

        for reduction_name in results.names:
            embedding = results[reduction_name]
            distance_matrix_low_dim = calculate_distance_matrix(embedding)
            distance_vector_low_dim = distance_matrix_low_dim[np.triu_indices(distance_matrix_low_dim.shape[0], k=1)]
            
            #spearman_p2p
            ranked_distance_vector_low_dim = distance_vector_low_dim.argsort().argsort()

            correlation, p_value = spearmanr(ranked_distance_vector_high_dim, ranked_distance_vector_low_dim)

            results.add_custom_evaluation_result(metric_name = "spearman_p2p",
                                                reduction_name = reduction_name, value = correlation)

            
            #emd_p2p
            emd = EM.EMD(distance_vector_high_dim, distance_vector_low_dim, normalization="minmax")

            results.add_custom_evaluation_result(metric_name = "emd_p2p",
                                                reduction_name = reduction_name, value = emd, reverse_ranking = True)
            

            #knn k=1% of datapoints
            embedding_neighbors = EM.build_annoy(results[reduction_name], k=math.ceil(len(self.data_handler.data) * 0.01))
            knn = EM.KNN(data_neighbors=original_neighbors, embedding_neighbors=embedding_neighbors)
            results.add_custom_evaluation_result(metric_name = "knn_0.01",
                                                reduction_name = reduction_name, value = knn)

            evals = results.evaluations

            #Cluster Concordance: ARI
            ari = evals["downstream"]["cluster concordance: ARI"][reduction_name]
            results.add_custom_evaluation_result(metric_name = "ari",
                                                reduction_name = reduction_name, value = ari)

            #Cluster Concordance: NMI
            nmi = evals["downstream"]["cluster concordance: NMI"][reduction_name]
            results.add_custom_evaluation_result(metric_name = "nmi",
                                                reduction_name = reduction_name, value = nmi)
            
        return results
        
    #UMAP
    def umap_best_param(self, n_neighbors_values, min_dist_values, log=True, plot=True, auto_cluster=True):
        reductions = {}
        runtime_dict = {}
        cluster_dict = {}
        counter = 1
        total_combs = str(len(n_neighbors_values) * len(min_dist_values) * 3)
        for n_neighbors in n_neighbors_values:
            for min_dist in min_dist_values:
                for i in range(3):
                    st = time.process_time()
                    embedding = dr.NonLinearMethods.UMAP(data=self.data_handler.data, out_dims=self.data_handler.out_dims, n_neighbors=n_neighbors, min_dist=min_dist)
                    et = time.process_time()
                    run_name = f"{n_neighbors}_{min_dist}({i})"
                    reductions[run_name] = embedding
                    runtime_dict[run_name] = et - st
                    if auto_cluster == False:
                        cluster_dict[run_name] = self.data_handler.cluster_custom(embedding, method="kmeans")
                    print("UMAP Parameter Combination: " + str(counter) + " out of " + total_combs)
                    counter += 1
        print("UMAP Parameter Tuning Finished!")
        results = dr.Reductions(reductions=reductions)
        results.add_evaluation_metadata(original_data=self.data_handler.data)
        if auto_cluster == False:
            results.add_evaluation_metadata(original_labels=self.data_handler.high_dim_clusters, embedding_labels=cluster_dict)
        results.evaluate(category = ["global", "local", "downstream"], auto_cluster = auto_cluster, n_clusters = self.data_handler.n_groups, verbose=False, normalize_pwd="minmax")
        #Custom-Metrics
        results = self.add_custom_evals(results)
        eval_df = DimRedBenchmarker.get_eval_df(results)
        if log:
            log_csv_path = self.log_folder_path + "/" + self.data_handler.dataset_name + "_log.csv"
            self.log_per_method_reduction("umap", results, runtime_dict, log_csv_path)

        if plot:
            DimRedBenchmarker.plot_param_tune_accuracy(eval_df=eval_df, metrics="all", dataset_name=self.data_handler.dataset_name,
                                                       out_dims=self.data_handler.out_dims, n_param_sets = 5,
                                                       save_folder=self.plot_folder_path, method="umap")

        #rank by custom metrics
        rank_dict = results.rank_dr_methods_custom()
        max_key = max(rank_dict, key=rank_dict.get)
        best_params = max_key.split("_")
        best_params[1] = best_params[1].split("(")[0]
        return {"n_neighbors": int(best_params[0]), "min_dist": float(best_params[1])}

    #tsne bh
    def tsne_bh_best_param(self, perplexity_values, log=True, plot=True, auto_cluster=True):
        reductions = {}
        runtime_dict = {}
        cluster_dict = {}
        counter = 1
        total_combs = str(len(perplexity_values))
        for perplexity in perplexity_values:
            st = time.process_time()
            embedding = dr.NonLinearMethods.sklearn_tsne(data=self.data_handler.data, out_dims=self.data_handler.out_dims, perplexity=perplexity)
            et = time.process_time()
            run_name = str(perplexity)
            reductions[run_name] = embedding
            runtime_dict[run_name] = et - st
            if auto_cluster == False:
                cluster_dict[run_name] = self.data_handler.cluster_custom(embedding, method="kmeans")
            print("tsne bh Parameter Combination: " + str(counter) + " out of " + total_combs)
            counter += 1
        print("tsne bh Parameter Tuning Finished!")
        results = dr.Reductions(reductions=reductions)
        results.add_evaluation_metadata(original_data=self.data_handler.data)
        if auto_cluster == False:
            results.add_evaluation_metadata(original_labels=self.data_handler.high_dim_clusters, embedding_labels=cluster_dict)
        results.evaluate(category = ["global", "local", "downstream"], auto_cluster = auto_cluster, n_clusters = self.data_handler.n_groups, verbose=False, normalize_pwd="minmax")
        #Custom-Metrics
        results = self.add_custom_evals(results)
        eval_df = DimRedBenchmarker.get_eval_df(results)
        if log:
            log_csv_path = self.log_folder_path + "/" + self.data_handler.dataset_name + "_log.csv"
            self.log_per_method_reduction("tsne_bh", results, runtime_dict, log_csv_path)

        if plot:
            DimRedBenchmarker.plot_param_tune_accuracy(eval_df=eval_df, metrics="all", dataset_name=self.data_handler.dataset_name,
                                                       out_dims=self.data_handler.out_dims, n_param_sets = 5,
                                                       save_folder=self.plot_folder_path, method="tsne_bh")


        rank_dict = results.rank_dr_methods_custom()
        max_key = max(rank_dict, key=rank_dict.get)
        return {"perplexity": float(max_key)}


    #phate
    def phate_best_param(self, knn_values, gamma_values, log=True, plot=True, auto_cluster=True):
        reductions = {}
        runtime_dict = {}
        cluster_dict = {}
        counter = 1
        total_combs = str(len(knn_values) * len(gamma_values))
        for knn in knn_values:
            for gamma in gamma_values:
                st = time.process_time()
                embedding = dr.NonLinearMethods.phate(data=self.data_handler.data, out_dims=self.data_handler.out_dims, knn=knn, gamma=gamma)
                et = time.process_time()
                run_name = str(knn) + "_" + str(gamma)
                reductions[run_name] = embedding
                runtime_dict[run_name] = et - st
                if auto_cluster == False:
                    cluster_dict[run_name] = self.data_handler.cluster_custom(embedding, method="kmeans")
                print("PHATE Parameter Combination: " + str(counter) + " out of " + total_combs)
                counter += 1
        print("PHATE Parameter Tuning Finished!")
        results = dr.Reductions(reductions=reductions)
        results.add_evaluation_metadata(original_data=self.data_handler.data)
        if auto_cluster == False:
            results.add_evaluation_metadata(original_labels=self.data_handler.high_dim_clusters, embedding_labels=cluster_dict)
        results.evaluate(category = ["global", "local", "downstream"], auto_cluster = auto_cluster, n_clusters = self.data_handler.n_groups, verbose=False, normalize_pwd="minmax")
        #Custom-Metrics
        results = self.add_custom_evals(results)
        eval_df = DimRedBenchmarker.get_eval_df(results)
        if log:
            log_csv_path = self.log_folder_path + "/" + self.data_handler.dataset_name + "_log.csv"
            self.log_per_method_reduction("phate", results, runtime_dict, log_csv_path)

        if plot:
            DimRedBenchmarker.plot_param_tune_accuracy(eval_df=eval_df, metrics="all", dataset_name=self.data_handler.dataset_name,
                                                       out_dims=self.data_handler.out_dims, n_param_sets = 5,
                                                       save_folder=self.plot_folder_path, method="phate")

        rank_dict = results.rank_dr_methods_custom()
        max_key = max(rank_dict, key=rank_dict.get)
        best_params = max_key.split("_")
        return {"knn": int(best_params[0]), "gamma": int(best_params[1])}


    #Isomap
    def isomap_best_param(self, n_neighbors_values, log=True, plot=True, auto_cluster=True):
        reductions = {}
        runtime_dict = {}
        cluster_dict = {}
        counter = 1
        total_combs = str(len(n_neighbors_values))
        for n_neighbors in n_neighbors_values:
            st = time.process_time()
            embedding = dr.NonLinearMethods.isomap(data=self.data_handler.data, out_dims=self.data_handler.out_dims, n_neighbors=n_neighbors)
            et = time.process_time()
            run_name = str(n_neighbors)
            reductions[run_name] = embedding
            runtime_dict[run_name] = et - st
            if auto_cluster == False:
                cluster_dict[run_name] = self.data_handler.cluster_custom(embedding, method="kmeans")
            print("Isomap Parameter Combination: " + str(counter) + " out of " + total_combs)
            counter += 1
        print("Isomap Parameter Tuning Finished!")
        results = dr.Reductions(reductions=reductions)
        results.add_evaluation_metadata(original_data=self.data_handler.data)
        if auto_cluster == False:
            results.add_evaluation_metadata(original_labels=self.data_handler.high_dim_clusters, embedding_labels=cluster_dict)
        results.evaluate(category = ["global", "local", "downstream"], auto_cluster = auto_cluster, n_clusters = self.data_handler.n_groups, verbose=False, normalize_pwd="minmax")
        #Custom-Metrics
        results = self.add_custom_evals(results)
        eval_df = DimRedBenchmarker.get_eval_df(results)
        if log:
            log_csv_path = self.log_folder_path + "/" + self.data_handler.dataset_name + "_log.csv"
            self.log_per_method_reduction("isomap", results, runtime_dict, log_csv_path)

        if plot:
            DimRedBenchmarker.plot_param_tune_accuracy(eval_df=eval_df, metrics="all", dataset_name=self.data_handler.dataset_name,
                                                       out_dims=self.data_handler.out_dims, n_param_sets = 5,
                                                       save_folder=self.plot_folder_path, method="isomap")

        rank_dict = results.rank_dr_methods_custom()
        max_key = max(rank_dict, key=rank_dict.get)
        return {"n_neighbors": int(max_key)}


    #MDS
    def mds_best_param(self, eps_values, log=True, plot=True, auto_cluster=True):
        reductions = {}
        runtime_dict = {}
        cluster_dict = {}
        counter = 1
        total_combs = str(len(eps_values))
        for eps in eps_values:
            st = time.process_time()
            embedding = dr.NonLinearMethods.MDS(data=self.data_handler.data, out_dims=self.data_handler.out_dims, eps=eps)
            et = time.process_time()
            run_name = str(eps)
            reductions[run_name] = embedding
            runtime_dict[run_name] = et - st
            if auto_cluster == False:
                cluster_dict[run_name] = self.data_handler.cluster_custom(embedding, method="kmeans")
            print("MDS Parameter Combination: " + str(counter) + " out of " + total_combs)
            counter += 1
        print("MDS Parameter Tuning Finished!")
        results = dr.Reductions(reductions=reductions)
        results.add_evaluation_metadata(original_data=self.data_handler.data)
        if auto_cluster == False:
            results.add_evaluation_metadata(original_labels=self.data_handler.high_dim_clusters, embedding_labels=cluster_dict)
        results.evaluate(category = ["global", "local", "downstream"], auto_cluster = auto_cluster, n_clusters = self.data_handler.n_groups, verbose=False, normalize_pwd="minmax")
        #Custom-Metrics
        results = self.add_custom_evals(results)
        eval_df = DimRedBenchmarker.get_eval_df(results)
        if log:
            log_csv_path = self.log_folder_path + "/" + self.data_handler.dataset_name + "_log.csv"
            self.log_per_method_reduction("mds", results, runtime_dict, log_csv_path)

        if plot:
            DimRedBenchmarker.plot_param_tune_accuracy(eval_df=eval_df, metrics="all", dataset_name=self.data_handler.dataset_name,
                                                       out_dims=self.data_handler.out_dims, n_param_sets = 5,
                                                       save_folder=self.plot_folder_path, method="mds")

        rank_dict = results.rank_dr_methods_custom()
        max_key = max(rank_dict, key=rank_dict.get)
        return {"eps": float(max_key)}

    #LLE
    def lle_best_param(self, n_neighbors_values, reg_values, log=True, plot=True, auto_cluster=True):
        reductions = {}
        runtime_dict = {}
        cluster_dict = {}
        counter = 1
        total_combs = str(len(n_neighbors_values) * len(reg_values))
        for n_neighbors in n_neighbors_values:
            for reg in reg_values:
                st = time.process_time()
                embedding = dr.NonLinearMethods.LLE(data=self.data_handler.data, out_dims=self.data_handler.out_dims, n_neighbors=n_neighbors, reg=reg, eigen_solver="dense")
                et = time.process_time()
                run_name = str(n_neighbors) + "_" + str(reg)
                reductions[run_name] = embedding
                runtime_dict[run_name] = et - st
                if auto_cluster == False:
                    cluster_dict[run_name] = self.data_handler.cluster_custom(embedding, method="kmeans")
                print("LLE Parameter Combination: " + str(counter) + " out of " + total_combs)
                counter += 1
        print("LLE Parameter Tuning Finished!")
        results = dr.Reductions(reductions=reductions)
        results.add_evaluation_metadata(original_data=self.data_handler.data)
        if auto_cluster == False:
            results.add_evaluation_metadata(original_labels=self.data_handler.high_dim_clusters, embedding_labels=cluster_dict)
        results.evaluate(category = ["global", "local", "downstream"], auto_cluster = auto_cluster, n_clusters = self.data_handler.n_groups, verbose=False, normalize_pwd="minmax")
        #Custom-Metrics
        results = self.add_custom_evals(results)
        eval_df = DimRedBenchmarker.get_eval_df(results)
        if log:
            log_csv_path = self.log_folder_path + "/" + self.data_handler.dataset_name + "_log.csv"
            self.log_per_method_reduction("lle", results, runtime_dict, log_csv_path)

        if plot:
            DimRedBenchmarker.plot_param_tune_accuracy(eval_df=eval_df, metrics="all", dataset_name=self.data_handler.dataset_name,
                                                       out_dims=self.data_handler.out_dims, n_param_sets = 5,
                                                       save_folder=self.plot_folder_path, method="lle")

        rank_dict = results.rank_dr_methods_custom()
        max_key = max(rank_dict, key=rank_dict.get)
        best_params = max_key.split("_")
        return {"n_neighbors": int(best_params[0]), "reg": float(best_params[1])}


    #KPCA (Poly)
    def kpca_poly_best_param(self, gamma_values, degree_values, log=True, plot=True, auto_cluster=True):
        reductions = {}
        runtime_dict = {}
        cluster_dict = {}
        counter = 1
        total_combs = str(len(gamma_values) * len(degree_values))
        for gamma in gamma_values:
            for degree in degree_values:
                st = time.process_time()
                embedding = dr.NonLinearMethods.kernelPCA(data=self.data_handler.data, out_dims=self.data_handler.out_dims, kernel="poly", gamma=gamma, degree=degree)
                et = time.process_time()
                run_name = str(gamma) + "_" + str(degree)
                reductions[run_name] = embedding
                runtime_dict[run_name] = et - st
                if auto_cluster == False:
                    cluster_dict[run_name] = self.data_handler.cluster_custom(embedding, method="kmeans")
                print("KPCA Poly Parameter Combination: " + str(counter) + " out of " + total_combs)
                counter += 1
        print("KPCA Poly Parameter Tuning Finished!")
        results = dr.Reductions(reductions=reductions)
        results.add_evaluation_metadata(original_data=self.data_handler.data)
        if auto_cluster == False:
            results.add_evaluation_metadata(original_labels=self.data_handler.high_dim_clusters, embedding_labels=cluster_dict)
        results.evaluate(category = ["global", "local", "downstream"], auto_cluster = auto_cluster, n_clusters = self.data_handler.n_groups, verbose=False, normalize_pwd="minmax")
        #Custom-Metrics
        results = self.add_custom_evals(results)
        eval_df = DimRedBenchmarker.get_eval_df(results)
        if log:
            log_csv_path = self.log_folder_path + "/" + self.data_handler.dataset_name + "_log.csv"
            self.log_per_method_reduction("kpca_poly", results, runtime_dict, log_csv_path)

        if plot:
            DimRedBenchmarker.plot_param_tune_accuracy(eval_df=eval_df, metrics="all", dataset_name=self.data_handler.dataset_name,
                                                       out_dims=self.data_handler.out_dims, n_param_sets = 5,
                                                       save_folder=self.plot_folder_path, method="kpca_poly")

        rank_dict = results.rank_dr_methods_custom()
        max_key = max(rank_dict, key=rank_dict.get)
        best_params = max_key.split("_")
        return {"gamma": float(best_params[0]), "degree": int(best_params[1])}

    #KPCA (rbf)
    def kpca_rbf_best_param(self, gamma_values, log=True, plot=True, auto_cluster=True):
        reductions = {}
        runtime_dict = {}
        cluster_dict = {}
        counter = 1
        total_combs = str(len(gamma_values))
        for gamma in gamma_values:
            st = time.process_time()
            embedding = dr.NonLinearMethods.kernelPCA(data=self.data_handler.data, out_dims=self.data_handler.out_dims, kernel="rbf", gamma=gamma)
            et = time.process_time()
            run_name = str(gamma)
            reductions[run_name] = embedding
            runtime_dict[run_name] = et - st
            if auto_cluster == False:
                cluster_dict[run_name] = self.data_handler.cluster_custom(embedding, method="kmeans")
            print("KPCA rbf Parameter Combination: " + str(counter) + " out of " + total_combs)
            counter += 1
        print("KPCA rbf Parameter Tuning Finished!")
        results = dr.Reductions(reductions=reductions)
        results.add_evaluation_metadata(original_data=self.data_handler.data)
        if auto_cluster == False:
            results.add_evaluation_metadata(original_labels=self.data_handler.high_dim_clusters, embedding_labels=cluster_dict)
        results.evaluate(category = ["global", "local", "downstream"], auto_cluster = auto_cluster, n_clusters = self.data_handler.n_groups, verbose=False, normalize_pwd="minmax")
        #Custom-Metrics
        results = self.add_custom_evals(results)
        eval_df = DimRedBenchmarker.get_eval_df(results)
        if log:
            log_csv_path = self.log_folder_path + "/" + self.data_handler.dataset_name + "_log.csv"
            self.log_per_method_reduction("kpca_rbf", results, runtime_dict, log_csv_path)

        if plot:
            DimRedBenchmarker.plot_param_tune_accuracy(eval_df=eval_df, metrics="all", dataset_name=self.data_handler.dataset_name,
                                                       out_dims=self.data_handler.out_dims, n_param_sets = 5,
                                                       save_folder=self.plot_folder_path, method="kpca_rbf")

        rank_dict = results.rank_dr_methods_custom()
        max_key = max(rank_dict, key=rank_dict.get)
        return {"gamma": float(max_key)}

    #Spectral
    def spectral_best_param(self, n_neighbors_values, gamma_values, log=True, plot=True, auto_cluster=True):
        reductions = {}
        runtime_dict = {}
        cluster_dict = {}
        counter = 1
        total_combs = str(len(n_neighbors_values) * len(gamma_values))
        for n_neighbors in n_neighbors_values:
            for gamma in gamma_values:
                st = time.process_time()
                embedding = dr.NonLinearMethods.spectral(data=self.data_handler.data, out_dims=self.data_handler.out_dims, n_neighbors=n_neighbors, gamma=gamma)
                et = time.process_time()
                run_name = str(n_neighbors) + "_" + str(gamma)
                reductions[run_name] = embedding
                runtime_dict[run_name] = et - st
                if auto_cluster == False:
                    cluster_dict[run_name] = self.data_handler.cluster_custom(embedding, method="kmeans")
                print("Spectral Parameter Combination: " + str(counter) + " out of " + total_combs)
                counter += 1
        print("Spectral Parameter Tuning Finished!")
        results = dr.Reductions(reductions=reductions)
        results.add_evaluation_metadata(original_data=self.data_handler.data)
        if auto_cluster == False:
            results.add_evaluation_metadata(original_labels=self.data_handler.high_dim_clusters, embedding_labels=cluster_dict)
        results.evaluate(category = ["global", "local", "downstream"], auto_cluster = auto_cluster, n_clusters = self.data_handler.n_groups, verbose=False, normalize_pwd="minmax")
        #Custom-Metrics
        results = self.add_custom_evals(results)
        eval_df = DimRedBenchmarker.get_eval_df(results)
        if log:
            log_csv_path = self.log_folder_path + "/" + self.data_handler.dataset_name + "_log.csv"
            self.log_per_method_reduction("spectral", results, runtime_dict, log_csv_path)

        if plot:
            DimRedBenchmarker.plot_param_tune_accuracy(eval_df=eval_df, metrics="all", dataset_name=self.data_handler.dataset_name,
                                                       out_dims=self.data_handler.out_dims, n_param_sets = 5,
                                                       save_folder=self.plot_folder_path, method="spectral")

        rank_dict = results.rank_dr_methods_custom()
        max_key = max(rank_dict, key=rank_dict.get)
        best_params = max_key.split("_")
        return {"n_neighbors": int(best_params[0]), "gamma": float(best_params[1])}
    
    #cimura
    def cimura_best_param(self, n_centroids_values, scaling_factor_values, log=True, plot=True, auto_cluster=True):
        reductions = {}
        runtime_dict = {}
        cluster_dict = {}
        counter = 1
        total_combs = str(len(n_centroids_values) * len(scaling_factor_values))

        cimura = CIMURA(n_components=self.data_handler.out_dims)
        high_dim_trained_h5_path = "/mnt/project/guerguerian/CIMURA/files/new_10k_isomap/new_10k_isomap_high_dim_trained.h5"
        low_dim_trained_h5_path = "/mnt/project/guerguerian/CIMURA/files/new_10k_isomap/new_10k_isomap_low_dim_trained_shifted.h5"
        centroids_csv_path = "/mnt/project/guerguerian/CIMURA/files/new_10k_isomap/new_10k_isomap_centroids.csv"
        cimura.load_training(high_dim_trained_h5_path, low_dim_trained_h5_path, centroids_csv_path)

        for n_centroids in n_centroids_values:
            for scaling_factor in scaling_factor_values:
                local_reducer = umap.UMAP(n_neighbors=25, min_dist=0.5, n_components=self.data_handler.out_dims)
                st = time.process_time()
                embedding, emb_ids = cimura.fit_transform_multi_clusterfile(self.data_handler.clusterfile, self.data_handler.h5_path, n_centroids=n_centroids, scaling_factor=scaling_factor, local_reducer=local_reducer)
                et = time.process_time()
                
                sort_order = {id_val: i for i, id_val in enumerate(self.data_handler.ids)}
                sort_index = np.array([sort_order[id_val] for id_val in emb_ids])
                embedding = embedding[np.argsort(sort_index)]
                
                run_name = f"{n_centroids}_{scaling_factor}"
                reductions[run_name] = embedding
                runtime_dict[run_name] = et - st
                if auto_cluster == False:
                    cluster_dict[run_name] = self.data_handler.cluster_custom(embedding, method="kmeans")
                print("cimura Parameter Combination: " + str(counter) + " out of " + total_combs)
                counter += 1
        print("cimura Parameter Tuning Finished!")
        results = dr.Reductions(reductions=reductions)
        results.add_evaluation_metadata(original_data=self.data_handler.data)
        if auto_cluster == False:
            results.add_evaluation_metadata(original_labels=self.data_handler.high_dim_clusters, embedding_labels=cluster_dict)
        results.evaluate(category = ["global", "local", "downstream"], auto_cluster = auto_cluster, n_clusters = self.data_handler.n_groups, verbose=False, normalize_pwd="minmax")
        #Custom-Metrics
        results = self.add_custom_evals(results)
        eval_df = DimRedBenchmarker.get_eval_df(results)
        if log:
            log_csv_path = self.log_folder_path + "/" + self.data_handler.dataset_name + "_log.csv"
            self.log_per_method_reduction("cimura", results, runtime_dict, log_csv_path)

        if plot:
            DimRedBenchmarker.plot_param_tune_accuracy(eval_df=eval_df, metrics="all", dataset_name=self.data_handler.dataset_name,
                                                       out_dims=self.data_handler.out_dims, n_param_sets = 5,
                                                       save_folder=self.plot_folder_path, method="cimura")

        rank_dict = results.rank_dr_methods_custom()
        max_key = max(rank_dict, key=rank_dict.get)
        best_params = max_key.split("_")
        return {"n_centroids": int(best_params[0]),
                "scaling_factor": float(best_params[1])}

    
    
    
    
    #Hyperparameter Tuning for each method on the given paramset
    def best_param_search(self, methods_parameters, auto_cluster):
        
        best_param_dicts = {}

        #cimura
        if "cimura" in methods_parameters.keys():
            print("cimura PARAMETER TUNING")
            n_centroids_values = methods_parameters["cimura"]["n_centroids"]
            scaling_factor_values = methods_parameters["cimura"]["scaling_factor"]
            cimura_best_param_dict = self.cimura_best_param(n_centroids_values, scaling_factor_values, auto_cluster=auto_cluster)
            print(cimura_best_param_dict)
            best_param_dicts["cimura"] = cimura_best_param_dict
            
        #UMAP
        if "umap" in methods_parameters.keys():
            print("UMAP PARAMETER TUNING")
            n_neighbors_values = methods_parameters["umap"]["n_neighbors_values"]
            min_dist_values = methods_parameters["umap"]["min_dist_values"]
            umap_best_param_dict = self.umap_best_param(n_neighbors_values, min_dist_values, auto_cluster=auto_cluster)
            print(umap_best_param_dict)
            best_param_dicts["umap"] = umap_best_param_dict

        #tsne bh
        if "tsne_bh" in methods_parameters.keys():
            print("tsne bh PARAMETER TUNING")
            perplexity_values = methods_parameters["tsne_bh"]["perplexity_values"]
            tsne_bh_best_param_dict = self.tsne_bh_best_param(perplexity_values, auto_cluster=auto_cluster)
            print(tsne_bh_best_param_dict)
            best_param_dicts["tsne_bh"] = tsne_bh_best_param_dict

        #PHATE
        if "phate" in methods_parameters.keys():
            print("PHATE PARAMETER TUNING")
            knn_values = methods_parameters["phate"]["knn_values"]
            gamma_values = methods_parameters["phate"]["gamma_values"]
            phate_best_param_dict = self.phate_best_param(knn_values=knn_values, gamma_values=gamma_values, auto_cluster=auto_cluster)
            print(phate_best_param_dict)
            best_param_dicts["phate"] = phate_best_param_dict

        #Isomap
        if "isomap" in methods_parameters.keys():
            print("Isomap PARAMETER TUNING")
            n_neighbors_values = methods_parameters["isomap"]["n_neighbors_values"]
            isomap_best_param_dict = self.isomap_best_param(n_neighbors_values, auto_cluster=auto_cluster)
            print(isomap_best_param_dict)
            best_param_dicts["isomap"] = isomap_best_param_dict

        #MDS
        if "mds" in methods_parameters.keys():
            print("MDS PARAMETER TUNING")
            eps_values = methods_parameters["mds"]["eps_values"]
            mds_best_param_dict = self.mds_best_param(eps_values, auto_cluster=auto_cluster)
            print(mds_best_param_dict)
            best_param_dicts["mds"] = mds_best_param_dict

        #LLE
        if "lle" in methods_parameters.keys():
            print("LLE PARAMETER TUNING")
            n_neighbors_values = methods_parameters["lle"]["n_neighbors_values"]
            reg_values = methods_parameters["lle"]["reg_values"]
            lle_best_param_dict = self.lle_best_param(n_neighbors_values, reg_values, auto_cluster=auto_cluster)
            print(lle_best_param_dict)
            best_param_dicts["lle"] = lle_best_param_dict


        #KPCA Poly
        if "kpca_poly" in methods_parameters.keys():
            print("KPCA Poly PARAMETER TUNING")
            gamma_values = methods_parameters["kpca_poly"]["gamma_values"]
            degree_values = methods_parameters["kpca_poly"]["degree_values"]
            kpca_poly_best_param_dict = self.kpca_poly_best_param(gamma_values, degree_values, auto_cluster=auto_cluster)
            print(kpca_poly_best_param_dict)
            best_param_dicts["kpca_poly"] = kpca_poly_best_param_dict

        #KPCA rbf
        if "kpca_rbf" in methods_parameters.keys():
            print("KPCA rbf PARAMETER TUNING")
            gamma_values = methods_parameters["kpca_rbf"]["gamma_values"]
            kpca_rbf_best_param_dict = self.kpca_rbf_best_param(gamma_values, auto_cluster=auto_cluster)
            print(kpca_rbf_best_param_dict)
            best_param_dicts["kpca_rbf"] = kpca_rbf_best_param_dict

        #Spectral
        if "spectral" in methods_parameters.keys():
            print("SPECTRAL PARAMETER TUNING")
            n_neighbors_values = methods_parameters["spectral"]["n_neighbors_values"]
            gamma_values = methods_parameters["spectral"]["gamma_values"]
            spectral_best_param_dict = self.spectral_best_param(n_neighbors_values, gamma_values, auto_cluster=auto_cluster)
            print(spectral_best_param_dict)
            best_param_dicts["spectral"] = spectral_best_param_dict


        #Additional untuneable Methods
        #tsne_fft
        if "tsne_fft" in methods_parameters.keys():
            print("tsne_fft, no tuning needed.")
            tsne_fft_best_param_dict = {"perp": methods_parameters["tsne_fft"]["perp"]}
            best_param_dicts["tsne_fft"] = tsne_fft_best_param_dict

        #pca
        if "pca" in methods_parameters.keys():
            print("pca, no tuning needed.")
            best_param_dicts["pca"] = ""
        
        #ica
        if "ica" in methods_parameters.keys():
            print("ica, no tuning needed.")
            best_param_dicts["ica"] = ""

        if "fa" in methods_parameters.keys():
            print("fa, no tuning needed.")
            best_param_dicts["fa"] = ""


        return best_param_dicts


    def log_per_method_reduction(self, method, reduction, runtime_dict, log_csv_path, method_map=False, method_map_params={}):
        #Date and time
        current_datetime = datetime.datetime.now()
        today_date = current_datetime.date().strftime("%Y-%m-%d")
        current_time = current_datetime.time().strftime("%H:%M:%S")
        map_col = "method" if method_map == True else "parameters"
        #Reduction
        #rank_dict = reduction.rank_dr_methods()
        #custom_rank_dict = reduction.rank_dr_methods_custom()

        #Create and fill df
        log_df = DimRedBenchmarker.get_eval_df(reduction)
        if method_map == False:
            log_df["parameters"] = log_df.index
            log_df.insert(0, "method", method)
        else:
            log_df["method"] = log_df.index
            log_df["parameters"] = log_df[map_col].map(method_map_params)
        log_df["runtime"] = log_df[map_col].map(runtime_dict)
        #log_df["rank"] = log_df[map_col].map(rank_dict)
        #log_df["rank_custom"] = log_df[map_col].map(custom_rank_dict)
        log_df.insert(0, "time", current_time)
        log_df.insert(0, "date", today_date)
        log_df.insert(0, "out_dims", self.data_handler.out_dims)
        
        log_df = log_df.reset_index(drop=True)
        log_df = log_df[["date", "time", "method", "parameters", "runtime", "out_dims", "spearman", "emd", "knn", "npe", "cluster reconstruction: silhouette", "cluster reconstruction: DBI", "cluster reconstruction: CHI", "cluster reconstruction: RF", "cluster concordance: ARI", "cluster concordance: NMI","spearman_p2p", "emd_p2p", "knn_0.01", "ari", "nmi",  "rank", "rank_custom"]]
        
        if os.path.exists(log_csv_path):
            log_df.to_csv(log_csv_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_csv_path, mode="a", header=True, index=False)

        print("logging of " + method + " on " + self.data_handler.dataset_name + " successful.")

    
    #Create one embedding for each specified method (with given parameters)
    def create_reduction_object(self, methods_param_dict, log=True, plot=True, auto_cluster=True):
        reductions = {}
        runtime_dict = {}
        param_dict = {}
        cluster_dict = {}
        #UMAP embedding
        if "umap" in methods_param_dict.keys():
            st = time.process_time()
            reductions["umap"] = dr.NonLinearMethods.UMAP(data=self.data_handler.data, out_dims=self.data_handler.out_dims,
                                                    n_neighbors=methods_param_dict["umap"]["n_neighbors"],
                                                    min_dist=methods_param_dict["umap"]["min_dist"])
            et = time.process_time()
            runtime_dict["umap"] = et - st
            if auto_cluster == False:
                cluster_dict["umap"] = self.data_handler.cluster_custom(reductions["umap"], method="kmeans")
            param_dict["umap"] = str(methods_param_dict["umap"]["n_neighbors"]) + "_" + str(methods_param_dict["umap"]["min_dist"])
            

        #tsne bh embedding
        if "tsne_bh" in methods_param_dict.keys():
            st = time.process_time()
            reductions["tsne_bh"] = dr.NonLinearMethods.sklearn_tsne(data=self.data_handler.data, out_dims=self.data_handler.out_dims,
                                                                perplexity=methods_param_dict["tsne_bh"]["perplexity"])
            et = time.process_time()
            runtime_dict["tsne_bh"] = et - st
            if auto_cluster == False:
                cluster_dict["tsne_bh"] = self.data_handler.cluster_custom(reductions["tsne_bh"], method="kmeans")
            param_dict["tsne_bh"] = str(methods_param_dict["tsne_bh"]["perplexity"])
            
        #tsne fft embedding
        if "tsne_fft" in methods_param_dict.keys():
            st = time.process_time()
            reductions["tsne_fft"] = dr.NonLinearMethods.open_tsne(data=self.data_handler.data, out_dims=self.data_handler.out_dims,
                                                                perp=methods_param_dict["tsne_fft"]["perp"])
            et = time.process_time()
            runtime_dict["tsne_fft"] = et - st
            if auto_cluster == False:
                cluster_dict["tsne_fft"] = self.data_handler.cluster_custom(reductions["tsne_fft"], method="kmeans")
            min_val = min(methods_param_dict["tsne_fft"]["perp"])
            max_val = max(methods_param_dict["tsne_fft"]["perp"])
            n_vals = len(methods_param_dict["tsne_fft"]["perp"])
            param_dict["tsne_fft"] = str(min_val) + "_" + str(max_val) + "_" + str(n_vals)
            

        #phate embedding
        if "phate" in methods_param_dict.keys():
            st = time.process_time()
            reductions["phate"] = dr.NonLinearMethods.phate(data=self.data_handler.data, out_dims=self.data_handler.out_dims,
                                                    knn=methods_param_dict["phate"]["knn"],
                                                    gamma=methods_param_dict["phate"]["gamma"])
            et = time.process_time()
            runtime_dict["phate"] = et - st
            if auto_cluster == False:
                cluster_dict["phate"] = self.data_handler.cluster_custom(reductions["phate"], method="kmeans")
            param_dict["phate"] = str(methods_param_dict["phate"]["knn"]) + "_" + str(methods_param_dict["phate"]["gamma"])

        #PCA embedding
        if "pca" in methods_param_dict.keys():
            st = time.process_time()
            reductions["pca"] = dr.LinearMethods.PCA(data=self.data_handler.data, out_dims=self.data_handler.out_dims)
            et = time.process_time()
            runtime_dict["pca"] = et - st
            if auto_cluster == False:
                cluster_dict["pca"] = self.data_handler.cluster_custom(reductions["pca"], method="kmeans")
            param_dict["pca"] = "-"

        #ICA embedding (not sure if parameter fitting is needed)
        if "ica" in methods_param_dict.keys():
            st = time.process_time()
            reductions["ica"] = dr.LinearMethods.ICA(data=self.data_handler.data, out_dims=self.data_handler.out_dims)
            et = time.process_time()
            runtime_dict["ica"] = et - st
            if auto_cluster == False:
                cluster_dict["ica"] = self.data_handler.cluster_custom(reductions["ica"], method="kmeans")
            param_dict["ica"] = "-"

        #FA embedding (not sure if parameter fitting is needed)
        if "fa" in methods_param_dict.keys():
            st = time.process_time()
            reductions["fa"] = dr.LinearMethods.FA(data=self.data_handler.data, out_dims=self.data_handler.out_dims)
            et = time.process_time()
            runtime_dict["fa"] = et - st
            if auto_cluster == False:
                cluster_dict["fa"] = self.data_handler.cluster_custom(reductions["fa"], method="kmeans")
            param_dict["fa"] = "-"

        #Isomap embedding
        if "isomap" in methods_param_dict.keys():
            st = time.process_time()
            reductions["isomap"] = dr.NonLinearMethods.isomap(data=self.data_handler.data, out_dims=self.data_handler.out_dims,
                                                            n_neighbors=methods_param_dict["isomap"]["n_neighbors"])
            et = time.process_time()
            runtime_dict["isomap"] = et - st
            if auto_cluster == False:
                cluster_dict["isomap"] = self.data_handler.cluster_custom(reductions["isomap"], method="kmeans")
            param_dict["isomap"] = str(methods_param_dict["isomap"]["n_neighbors"])

        #MDS embedding
        if "mds" in methods_param_dict.keys():
            st = time.process_time()
            reductions["mds"] = dr.NonLinearMethods.MDS(data=self.data_handler.data, out_dims=self.data_handler.out_dims,
                                                        eps=methods_param_dict["mds"]["eps"])
            et = time.process_time()
            runtime_dict["mds"] = et - st
            if auto_cluster == False:
                cluster_dict["mds"] = self.data_handler.cluster_custom(reductions["mds"], method="kmeans")
            param_dict["mds"] = str(methods_param_dict["mds"]["eps"])


        #LLE embedding
        if "lle" in methods_param_dict.keys():
            st = time.process_time()
            reductions["lle"] = dr.NonLinearMethods.LLE(data=self.data_handler.data, out_dims=self.data_handler.out_dims,
                                                    n_neighbors=methods_param_dict["lle"]["n_neighbors"],
                                                    reg=methods_param_dict["lle"]["reg"], eigen_solver="dense")
            et = time.process_time()
            runtime_dict["lle"] = et - st
            if auto_cluster == False:
                cluster_dict["lle"] = self.data_handler.cluster_custom(reductions["lle"], method="kmeans")
            param_dict["lle"] = str(methods_param_dict["lle"]["n_neighbors"]) + "_" + str(methods_param_dict["lle"]["reg"])

        #KPCA (Poly) embedding
        if "kpca_poly" in methods_param_dict.keys():
            st = time.process_time()
            reductions["kpca_poly"] = dr.NonLinearMethods.kernelPCA(data=self.data_handler.data, out_dims=self.data_handler.out_dims, kernel="poly",
                                                    gamma=methods_param_dict["kpca_poly"]["gamma"],
                                                    degree=methods_param_dict["kpca_poly"]["degree"])
            et = time.process_time()
            runtime_dict["kpca_poly"] = et - st
            if auto_cluster == False:
                cluster_dict["kpca_poly"] = self.data_handler.cluster_custom(reductions["kpca_poly"], method="kmeans")
            param_dict["kpca_poly"] = str(methods_param_dict["kpca_poly"]["gamma"]) + "_" + str(methods_param_dict["kpca_poly"]["degree"])

        #KPCA (rbf) embedding
        if "kpca_rbf" in methods_param_dict.keys():
            st = time.process_time()
            reductions["kpca_rbf"] = dr.NonLinearMethods.kernelPCA(data=self.data_handler.data, out_dims=self.data_handler.out_dims, kernel="rbf",
                                                    gamma=methods_param_dict["kpca_rbf"]["gamma"])
            et = time.process_time()
            runtime_dict["kpca_rbf"] = et - st
            if auto_cluster == False:
                cluster_dict["kpca_rbf"] = self.data_handler.cluster_custom(reductions["kpca_rbf"], method="kmeans")
            param_dict["kpca_rbf"] = str(methods_param_dict["kpca_rbf"]["gamma"])


        #Spectral embedding
        if "spectral" in methods_param_dict.keys():
            st = time.process_time()
            reductions["spectral"] = dr.NonLinearMethods.spectral(data=self.data_handler.data, out_dims=self.data_handler.out_dims,
                                                    n_neighbors=methods_param_dict["spectral"]["n_neighbors"],
                                                    gamma=methods_param_dict["spectral"]["gamma"])
            et = time.process_time()
            runtime_dict["spectral"] = et - st
            if auto_cluster == False:
                cluster_dict["spectral"] = self.data_handler.cluster_custom(reductions["spectral"], method="kmeans")
            param_dict["spectral"] = str(methods_param_dict["spectral"]["n_neighbors"]) + "_" + str(methods_param_dict["spectral"]["gamma"])

        #cimura embedding
        if "cimura" in methods_param_dict.keys():
            cimura = CIMURA(n_components=self.data_handler.out_dims)
            high_dim_trained_h5_path = "/mnt/project/guerguerian/CIMURA/files/new_10k_isomap/new_10k_isomap_high_dim_trained.h5"
            low_dim_trained_h5_path = "/mnt/project/guerguerian/CIMURA/files/new_10k_isomap/new_10k_isomap_low_dim_trained_shifted.h5"
            centroids_csv_path = "/mnt/project/guerguerian/CIMURA/files/new_10k_isomap/new_10k_isomap_centroids.csv"
            cimura.load_training(high_dim_trained_h5_path, low_dim_trained_h5_path, centroids_csv_path)
            local_reducer = umap.UMAP(n_neighbors=25, min_dist=0.5, n_components=self.data_handler.out_dims)
            st = time.process_time()
            embedding, emb_ids = cimura.fit_transform_multi_clusterfile(self.data_handler.clusterfile, self.data_handler.h5_path, n_centroids=methods_param_dict["cimura"]["n_centroids"], scaling_factor=methods_param_dict["cimura"]["scaling_factor"], local_reducer=local_reducer)
            et = time.process_time()
            
            sort_order = {id_val: i for i, id_val in enumerate(self.data_handler.ids)}
            sort_index = np.array([sort_order[id_val] for id_val in emb_ids])
            embedding = embedding[np.argsort(sort_index)]

            reductions["cimura"] = embedding

            runtime_dict["cimura"] = et - st
            if auto_cluster == False:
                cluster_dict["cimura"] = self.data_handler.cluster_custom(reductions["cimura"], method="kmeans")
            param_dict["cimura"] = str(methods_param_dict["cimura"]["n_centroids"]) + "_" + str(methods_param_dict["cimura"]["scaling_factor"])


        #Create Reductions Instance
        results = dr.Reductions(reductions=reductions)
        results.add_evaluation_metadata(original_data=self.data_handler.data)
        if auto_cluster == False:
            results.add_evaluation_metadata(original_labels=self.data_handler.high_dim_clusters, embedding_labels=cluster_dict)
        results.evaluate(category = ["global", "local", "downstream"], auto_cluster = auto_cluster, n_clusters = self.data_handler.n_groups, verbose=False, normalize_pwd="minmax")
        #Custom-Metrics
        results = self.add_custom_evals(results)
        if log:
            log_csv_path = self.log_folder_path + "/" + self.data_handler.dataset_name + "_single_param_log.csv"
            self.log_per_method_reduction(method="method_set", reduction=results, runtime_dict=runtime_dict,
                                          log_csv_path=log_csv_path, method_map=True, method_map_params=param_dict)
            for id, labels in cluster_dict.items():
                self.data_handler.save_ids_and_clusters(self.data_handler.ids, labels, self.data_handler.id_path, "low_dim", id)
        if plot:
            eval_df = DimRedBenchmarker.get_eval_df(results)
            DimRedBenchmarker.plot_reduction_accuracy(evals_df=eval_df, show=True, save=True, metrics="all", dataset_name=self.data_handler.dataset_name, single_param_dict=methods_param_dict, out_dims=self.data_handler.out_dims, save_folder=self.plot_folder_path)

            
        return results
    
#RUN

#Usage for multiple Datasets

###SETTINGS###

#SEE READ ME FOR THE REQUIRED FOLDER STRUCTURE
log_folder_path = r"path_to_log_folder"
plot_folder_path = r"path_to_plot_folder"
embedding_folder_path = r"path_to_embedding_folder"
id_path = r"path_to_id_folder"


#dataset_name: dataset_h5_path
h5_dict = {
    "dataset_1": "path_to_dataset.h5"
}

#Parameters to be tested
methods_parameters = {
    #Only tuneable methods
    "cimura": {
        "n_centroids": np.linspace(1, 30, 29).astype(int),
        "scaling_factor": np.array([2.0, 1.8, 1.5, 1.2, 1.0, 0.8, 0.5, 0.3, 0.1, 0.008, 0.005, 0.003, 0.001, 0.0001])
    },
    "umap": {
        "n_neighbors_values": np.linspace(5, 50, 45).astype(int),
        "min_dist_values": np.round(np.linspace(0, 0.99, 20), 3) 
    },
    "tsne_bh":{
        "perplexity_values": np.round(np.linspace(5, 50, 50), 3)
    },
    "phate":{
        "knn_values": np.linspace(5, 30, 25).astype(int),
        "gamma_values": np.linspace(0, 1, 2).astype(int)
    },
    "isomap":{
        "n_neighbors_values": np.linspace(2, 50, 48).astype(int)
    },
    "mds":{
        "eps_values": np.linspace(1e-9, 1e-3, 10)
    },
    "lle":{
        "n_neighbors_values": np.linspace(5, 100, 50).astype(int),
        "reg_values": np.round(np.linspace(1e-3, 1e-1, 10), 3)
    },
    "kpca_poly": {
        "gamma_values": np.round(np.linspace(0.01, 10, 30), 3),
        "degree_values": np.linspace(2, 5, 3).astype(int)
    },
    "kpca_rbf": {
        "gamma_values": np.round(np.linspace(0.01, 10, 50), 3)
    },
    "spectral": {
        "n_neighbors_values": np.linspace(5, 100, 50).astype(int),
        "gamma_values": np.round(np.linspace(0.01, 10, 10), 3)
    },
    "pca": {},
    "ica": {},
    "fa": {}
}

out_dims = 2

#Number of groups in the Datasets
ds_n_groups = {
    "KP3_esm2": 3
}

#CSV File with group clusterings
#cluster,id
clusterfiles = {
    "KP3_esm2": "path_to_cluster_csv.csv"
}



###SETTINGS END###

#save important infos and results of all datasets
ds_reduction_dict = {}
ds_best_param_dict = {}
ds_eval_df_dict = {}
ds_eval_rank_df_dict = {}
ds_eval_rank_class_df_dict = {}

for ds_name, ds_h5_path in h5_dict.items():
    dre = DimRedEmbedder(DimRedDataHandler(h5_path=ds_h5_path, n_groups=ds_n_groups[ds_name],
                                           out_dims=out_dims, dataset_name=ds_name, id_path=id_path, clusterfile=clusterfiles[ds_name]),
                        log_folder_path=log_folder_path, plot_folder_path=plot_folder_path)
    
    best_param_dict = dre.best_param_search(methods_parameters, auto_cluster=False)
    
    results = dre.create_reduction_object(best_param_dict, auto_cluster=False)
    

    ds_reduction_dict[ds_name] = results
    ds_best_param_dict[ds_name] = best_param_dict




#Save best embeddings
current_datetime = datetime.datetime.now()

formatted_date = current_datetime.strftime("%Y-%m-%d")
formatted_time = current_datetime.strftime("%H-%M-%S")

for dataset, result in ds_reduction_dict.items():
    result.save_all_reductions(save_dir=f"{embedding_folder_path}/{formatted_date}_{formatted_time}_{dataset}_", delimiter=",")

    with open(f"{embedding_folder_path}/{formatted_date}_{formatted_time}_{dataset}_annotation.txt", "w") as file:
            file.write(str(ds_best_param_dict[dataset]))
            file.write("\n")
            file.write(f"out_dims: {out_dims}")