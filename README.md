![image](https://github.com/danigugu/CIMURA/assets/159900656/90d6ee5f-0bf5-4221-b0c2-71dc3ad2dbb7)

# CIMURA
CIMURA is a Dimensionality Reduction Tool, specialized on visualizing Protein Datasets by projecting protein Langue Model embeddings.


## Getting Started
The currently easiest way to use CIMURA is the Google Colab Page. It uses precomputed matrices of reference proteins.
These have to be loaded into Colab before running. Be aware that this part might take a while.
This time can be used to upload your own data into Colab.
After, loading the training data, CIMURA can be used repeatedly without further delay.

- [CIMURA Google Colab](https://colab.research.google.com/drive/16_GBTcZ2jmi87vjuXUqK-NkQ1kR3KLN1?usp=sharing)

Make sure to follow the instructions.

## Usage
CIMURA is trained on a large protein dataset. It works best when subsetting the input data into neighborhoods and projecting them individually.
For this reason, a csv-file containing cluster labels for each protein in needed.

### Option 1: Provide your own clustering file:
Input:  
- h5 of pLM embeddings
- csv of cluster labels

CSV-Format:

cluster,id
cluster1,prot1
cluster1,prot2
cluster2,prot3

Make sure that the ids in the csv file match the dataset names in the h5 file.

Output:  -CIMURA output csv file


### Option 2: Compute cluster file using KMeans
Input:  -h5 of pLM embeddings

CIMURA will compute clusters based on the high-dimensional h5 data and use this to run the Dimensionality Reduction.

Output:  -CIMURA output csv file

