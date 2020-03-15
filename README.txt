Steps to run the code:
1. Code link: https://github.com/tri47/unsupervised_learning
2. Set up the environment with the libraries in enviroment.yml file (using conda or pip)
The libraries include:
- python=3.7.2
- matplotlib=3.0.3
- numpy=1.16.2
- pandas=0.24.1
- seaborn=0.10.0
- mlrose==1.3.0
3. Run the program to run the experiments and produce the graphs for 2 problems:
Syntax:
    python unsupervised_learning.py --dataset="DATANAME" --task="TASKNAME"

    DATANAME options:
        "tree": the tree dataset
        "bank":  the bank dataset
    TASKNAME options:
        "choose_k": run experiements to find optimal number of clusters.
        "cluster": run 2 x clustering algorithms (K-means and EM).
        "pca": run PCA on the choosen dataset and generate experiement results.
        "ica": run ICA and generate experiement results.
        "rp": run random projection and generate experiement results.
        "feature": run decision tree feature selection method and generate experiement results.
    The options below only works for tree dataset:
        "NN": Run neural network on projected data .
        "NN_clustering_KM": run neural network with KM clustering as an additional feature.
        "NN_clustering_EM": run neural network with EM clustering as an additional feature.
        "NN_clustering_KM_without_org": run neural network with KM clustering as the only feature.
        "NN_clustering_EM_without_org": run neural network with KM clustering as the only feature.

Examples:    
Run PCA For bank marketing problem:
    python unsupervised_learning.py  --dataset="bank" --task="pca"
Run NN on projected data for diseased tree problem
    python supervised_learning.py  --dataset="tree" --task="NN"

4. The data sets are already included. They were obtained from:

1. http://archive.ics.uci.edu/ml/datasets/Bank+Marketing
2. http://archive.ics.uci.edu/ml/datasets/wilt

Reference:
1.	Moro, S, Cortez, P. and Rita, P.. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014. 
2.	Johnson, B., Tateishi, R., Hoan, N., 2013. A hybrid pansharpening approach and multiscale object-based image analysis for mapping diseased pine and oak trees. International Journal of Remote Sensing, 34 (20), 6969-6982.
