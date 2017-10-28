# HDDR-Viz
High-dimensional data reordering and visualization framework

The framework reorder and visualize numeric high-dimensional datasets. <br />
Input: numeric data (csv-file) <br />
Output: reordered data (csv-file), visualization image (png-file) <br />
Visualization methods: parallel coordinates, scatter plots matrix, heatmap <br />
Reordering methods: EM-ordering (using Euclidean or Manhattan distance), TSP-means, Lin-Kernighan heuristic, HC-olo, t-SNE, MDS, random shuffle

# Prerequisites
Matlab Bio-informatics toolbox for HC-olo data reordering. <br />
R Studio package Scagnostics for Scagnostics metric calculation. <br />
Python packages: NumPy, SciPy, Scikit-learn, Spectral, Seaborn, Matplotlib, Anytree, Pandas, PIL, Matlab.engine, Rpy2

# Configuration
Provide path to LKH and Scagnostics implementations as well as path to the directory containing datasets in configuration file 'config.ini':

[libs] <br />
lkh=/home/libs/lkh.exe <br />
scags=/home/libs/get_scag.r <br />
[datasets] <br />
directory=/home/data/datasets

# User parameters
User can define parameters in a json-file. <br />
Possible options:<br />
```json
{
    "data_source_type':{"file","load"}, 
    "viz":["PC","heatmap","SPLOM","scaledPC"],
    "ordering":["original", "random2, "MDS","LK","EM","HColo","TSNE","EMmanhattan"],
    "ordering_direction":["rows","cols"]
}
```

Example json file: <br />
```json
{
    "header_column": false, 
    "delimiter": ",", 
    "data_source": "iris", 
    "ordering_direction": "rows",
    "class_column": "species", 
    "header": true, 
    "ordering_methods": ["random", "MDS", "TSNE", "HColo", "EM", "TSPmeans", "LK"], 
    "viz_method": "heatmap", 
    "output_name": "/home/output/demo/iris_heatmap/iris", 
    "data_source_type": "load"
}  
```

#Launch 
python main.py -parameters demo.json
