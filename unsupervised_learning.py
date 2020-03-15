import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import time
import mlrose

from pandas.plotting import parallel_coordinates
from scipy.stats import kurtosis, skew

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, Normalizer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate, train_test_split
from sklearn.decomposition import PCA, FastICA
from sklearn import random_projection

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, f1_score, silhouette_samples, silhouette_score

print('Start Unsupervised Learning ....')


def return_stratified_kcv_results(clf, x_data, y_data, verbose = False, last_curve = False):
    y_data = y_data.to_list()
    y_data = np.array([y_data]).transpose()

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    train_scores, test_scores, train_accuracys, test_accuracys = [], [], [], []
    train_times, test_times = [], []
    curves = []
    for train_index, test_index in skf.split(x_data, y_data):
        print('a CV')
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]       

        start_time = time.time()
        results = clf.fit(x_train, y_train)
        train_times.append(time.time()-start_time)
        y_train_pred = clf.predict(x_train)
        start_time = time.time()
        y_test_pred = clf.predict(x_test)
        test_times.append(time.time()-start_time)
       
        a_curve = results.fitness_curve
        curves.append(a_curve)

        a = np.concatenate([y_train,y_train_pred],axis=1)
        train_score =f1_score(y_train, y_train_pred) 
        test_score =f1_score(y_test, y_test_pred) 
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        train_scores.append(train_score)
        test_scores.append(test_score)
        train_accuracys.append(train_accuracy)
        test_accuracys.append(test_accuracy)

    if last_curve:
        curves = curves[-1]
    else:
        curves = np.array(curves)
        curves = curves.mean(axis=0)
        print(np.shape(curves))
    train_scores = np.array(train_scores)
    test_scores = np.array(test_scores)
    train_accuracys = np.array(train_accuracys)
    test_accuracys = np.array(test_accuracys)
    train_times = np.array(train_times)
    test_times = np.array(test_times)     

    return curves, train_scores.mean(), test_scores.mean(), train_accuracys.mean(), test_accuracys.mean(),train_times.mean(),test_times.mean()


# Experiment to find optimal value of K
def plot_silhouette_test(x_data, name):
    ks = [i for i in range(2,10)]
    km_results, em_results = [], []
    for i in ks:
        print(i)
        km = KMeans(n_clusters=i,n_init=30,max_iter=300, random_state = 100)
        y_km = km.fit_predict(x_data)
        s_score = silhouette_score(x_data, y_km, metric='euclidean')
        km_results.append(s_score)

        em = GaussianMixture(n_components = i)
        y_em = em.fit_predict(x_data)
        s_score = silhouette_score(x_data, y_em, metric='euclidean')
        em_results.append(s_score)        

    plt.plot(ks, km_results, marker='o')
    plt.plot(ks, em_results, marker='*')
    plt.xlabel('Clusters', fontsize =12)
    plt.ylabel('Silhouette Score', fontsize =12)
    plt.legend(['K-means', 'Expectation Maximization'],fontsize=12)
    plt.title(name, fontsize=12)
    plt.xticks(ks)
    plt.tight_layout()
    plt.show()


def plot_sse_test(x_data, name):
    ks = [i for i in range(2,14)]
    km_sse = []
    for i in ks:
        print(i)
        km = KMeans(n_clusters=i,n_init=30,max_iter=300, random_state = 100)
        y_km = km.fit_predict(x_data)
        score= km.inertia_/(len(y_km))
        km_sse.append(score)

    plt.plot(ks, km_sse, marker='o')
    plt.xlabel('Clusters', fontsize =12)
    plt.ylabel('SSE', fontsize =12)
    plt.title(name, fontsize=12)
    plt.xticks(ks)
    plt.tight_layout()
    plt.show()

def plot_bic_test(x_data, name):
    ks = [i for i in range(2,14)]
    scores = []
    for i in ks:
        print(i)
        em = GaussianMixture(n_components = i, covariance_type='full')
        y_em = em.fit_predict(x_data)
        score= em.aic(x_data)
        scores.append(score)

    plt.plot(ks, scores, marker='o')
    plt.xlabel('Clusters', fontsize =12)
    plt.ylabel('BIC score', fontsize =12)
    plt.title(name, fontsize=12)
    plt.xticks(ks)
    plt.tight_layout()
    plt.show()


def plot_clusters(df, sizes, class_col, title):
    columns = list(df.columns.values)
    columns.remove(class_col)
    fig, axes = plt.subplots(nrows=sizes[0], ncols=sizes[1])
    fig.tight_layout()
    col_ix = 0
    for ax_col, col_name in zip(axes.flatten(), columns):
        print(col_ix)
        sb.stripplot(x = class_col, y = col_name, data = df, jitter = 0.3, ax=ax_col, alpha=0.2) 
       # sb.catplot(x = class_col, y = col_name, kind='box', data= df)
        #sb.violinplot(x = class_col, y = col_name, data = df,  ax=ax_col, alpha=0.2) 

        col_ix += 1
    fig.suptitle(title, fontsize=12)
    plt.show()

def plot_clusters_num(df, sizes, class_col,title):
    df.sort_values(class_col,inplace=True)
    columns = list(df.columns.values)
    columns.remove(class_col)
    fig, axes = plt.subplots(nrows=sizes[0], ncols=sizes[1])
    fig.tight_layout() 
    class_0 = df[df[class_col] == 0]
    class_1 = df[df[class_col] == 1]
    col_ix = 0
    for ax_col, col_name in zip(axes.flatten(), columns):
        sb.distplot( df[df[class_col]==0][col_name] , color="skyblue", label="1", ax=ax_col,vertical=True)
        sb.distplot( df[df[class_col]==1][col_name] , color="red", label="0", ax=ax_col,vertical=True)
    fig.suptitle(title)
    plt.show()

def plot_clusters_cat(df, sizes, class_col):
    df.sort_values(class_col,inplace=True)
    columns = list(df.columns.values)
    columns.remove(class_col)
    fig, axes = plt.subplots(nrows=sizes[0], ncols=sizes[1])
    class_0 = df[df[class_col] == 0]
    class_1 = df[df[class_col] == 1]
    col_ix = 0

    for ax_col, col_name in zip(axes.flatten(), columns):
        print(df[df[class_col]==0][col_name] )
        class_0[col_name].value_counts().plot(kind='line',ax=ax_col, color = 'skyblue', alpha=1)
        class_1[col_name].value_counts().plot(kind='line',ax=ax_col, color='red', alpha=1)
      #  sb.distplot( class_1[col_name] , color="red", label="Sepal Width")
    fig.tight_layout() 

    plt.show()

def generate_clusters(alg,x_data, y_data,columns, title, sizes, class_col, x_data_org=None,type=None):
    if alg == 'KM':
        print('using KM')
        km = KMeans(n_clusters=2,n_init=30,max_iter=300,random_state=100)
    elif alg == 'EM':
        print('unsing EM')
        km = GaussianMixture(n_components = 2)
    y_km = km.fit_predict(x_data)
    print('Adjusted rand score: ', adjusted_rand_score(y_data,y_km.tolist()))
    print('class distribution ')
    y_km_array = np.array(y_km)
    unique, counts = np.unique(y_km_array, return_counts=True)
    print(np.asarray((unique, counts)).T)

    y_km = np.array([y_km]).transpose()
    print('x org: ', x_data_org)
    if x_data_org is None:
        x_data_cluster = np.concatenate((y_km, x_data),axis=1)
    else:
        x_data_cluster = np.concatenate((x_data_org,y_km),axis=1)
    df = pd.DataFrame(data=x_data_cluster,columns = columns)    
    df[class_col] = df[class_col].astype(int)
    if type == 'num':
        plot_clusters_num(df, sizes, class_col,title)
    else:
        plot_clusters(df, sizes, class_col,title)

#*****************************#
def main(tree_k=0,bank_k=0,tree_cluster=0,bank_cluster=0, \
      tree_pca=0, bank_pca=0, tree_ica=0, bank_ica=0,tree_rp=0,bank_rp=0,\
      tree_feature=0, bank_feature=0, tree_NN=0, NN_KM=0, NN_EM=0, NN_without_org=0
    ):

    bank_NN = 0

    # PREPROCESS WILT DATA
    data = pd.read_csv('wilt_full.csv')
    data['class'].replace(['n'],0,inplace=True)
    data['class'].replace(['w'],1,inplace=True)
    x_data = data.loc[:, data.columns != 'class']
    y_data = data.loc[:,'class']
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)
    columns = list(data.columns.values)
    random_state = 100
    # Hold out test set for final performance measure
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=random_state, shuffle=True, stratify=y_data)
    
    if tree_k:
        plot_silhouette_test(x_train,'Silhouette score for Diseased Tree dataset')
        plot_sse_test(x_data,'Sum Squared Errors (K-means) for Diseased Tree dataset')
        plot_bic_test(x_data,'BIC score (Expectation Maximization) for Diseased Tree dataset')  

    if tree_cluster: 
        # PLOT CLUSTER FOR K-MEANS 
        generate_clusters('KM',x_data,y_data,columns,'K-means cluster scatter plots for each attribute',[1,5],'class')
        # PLOT CLUSTER FOR EM
        generate_clusters('EM',x_data,y_data,columns, 'EM cluster scatter plots for each attribute',[1,5],'class')
        # PLOT CLUSTER FOR GROUND TRUTH
        plot_clusters(data,[1,5],'class', 'Ground truth cluster scatter plots for each attribute')

    if tree_pca:
        print(x_data.shape)
        transformer = PCA(n_components=2)
        x_pca = transformer.fit_transform(x_data)
        eigen_vals = transformer.explained_variance_
        print(x_pca.shape)
        proj = transformer.inverse_transform(x_pca)
        loss = ((x_data - proj) ** 2).mean()
        print('PCA loss is: ', loss)

        # Sebastian Raschka, Vahid Mirjalili - Python Machine Learning_ Machine Learning and Deep Learning with Python, scikit-learn, and TensorFlow 2
        total_eigen = sum(eigen_vals)
        var_exp = [(i/total_eigen) for i in sorted(eigen_vals,reverse=True)]
        cum_var_exp = np.cumsum(var_exp)
        plt.bar(range(1,3),var_exp, align='center',label='individual explained variance')
        plt.step(range(1,3), cum_var_exp, where ='mid', label ='Cummulative explained variance',color='green')
        plt.xlabel('Principal component index')
        plt.ylabel('Explained variance ratio')
        plt.tight_layout()
        plt.show()
        columns = ['class','principle component 1', 'principle component 2']
        generate_clusters('KM',x_pca, y_data,columns, 'Cluster dist. plots for each PCA component (K-means)', [1,2],'class',type='num')
        generate_clusters('EM',x_pca, y_data,columns, 'Cluster dist. plots for each PCA component (EM)', [1,2],'class',type='num')

    if tree_ica:
        kurts = []
        comps = [i for i in range(1,6)]
        for i in comps:
            transformer = FastICA(n_components=i)
            x_ICA = transformer.fit_transform(x_data)    
            kurt = kurtosis(x_ICA).mean()
            print(kurt)
            kurts.append(kurt)

        plt.plot(comps,kurts)
        plt.xlabel('Components')
        plt.ylabel('Kurtosis')
        plt.title('Kurtosis plot for ICA (Tree)')
        plt.xticks(comps)
        plt.show()
        
        transformer = FastICA(n_components=2)
        x_ICA = transformer.fit_transform(x_data)    
    #    mu = np.mean(x_data, axis=0)
    #  print(x_RP.shape)
    #  print(transformer.mixing_)
    #   proj2 = np.linalg.lstsq(x_RP.T, transformer.components_)[0]
    #  proj2 = x_RP.dot(transformer.components_) + mu
        
        proj = transformer.inverse_transform(x_ICA)
        loss = ((x_data - proj) ** 2).mean()
        print('ICA loss is: ', loss)
        columns = ['class','Independent component 1', 'Idependent component 2']
        generate_clusters('KM',x_ICA, y_data,columns, 'Cluster dist. plots for eachICA component (K-means)', [1,2],'class',type='num')
        generate_clusters('EM',x_ICA, y_data,columns, 'Cluster dist. plots for each ICA component (EM)', [1,2],'class',type='num')


        
    if tree_rp:
        losses, kurts = [], []
        comps = [i for i in range(2,6)]
        for i in comps:
            transformer = random_projection.GaussianRandomProjection(n_components=i)
            mu = np.mean(x_data, axis=0)
            x_RP = transformer.fit_transform(x_data)
            t_matrix = transformer.components_

            proj = np.linalg.lstsq(x_RP.T, t_matrix)[0] + mu
            loss = ((x_data - proj) ** 2).mean()
            kurt = kurtosis(x_RP).mean()
            kurts.append(kurt)
            losses.append(loss)
        fig = plt.figure(1)    
        ax = fig.add_subplot(121)
        ax.plot(comps,kurts)
        ax.set(xlabel='Components', ylabel='Kurtosis', title='Kurtosis plot for RP (Tree)',xticks=comps)

        ax = fig.add_subplot(122)
        ax.plot(comps,losses)
        ax.set(xlabel='Components', ylabel='Loss', title='Loss plot for RP (Tree)',xticks=comps)

        losses, kurts = [], []

        comps = range(1,11)
        for i in comps:
            transformer = random_projection.GaussianRandomProjection(n_components=2)
            mu = np.mean(x_data, axis=0)
            x_RP = transformer.fit_transform(x_data)
            t_matrix = transformer.components_
            proj = np.linalg.lstsq(x_RP.T, t_matrix)[0] + mu
            loss = ((x_data - proj) ** 2).mean()
            kurt = kurtosis(x_RP).mean()
            kurts.append(kurt)
            losses.append(loss)
        fig = plt.figure(2)    
        ax = fig.add_subplot(121)
        ax.plot(comps,kurts)
        ax.set(xlabel='Run index', ylabel='Kurtosis', title='Kurtosis plot for RP (Tree)',xticks=comps)
        ax = fig.add_subplot(122)
        ax.plot(comps,losses)
        ax.set(xlabel='Run index', ylabel='Loss', title='Loss plot for RP (Tree)',xticks=comps)

        plt.show()

        transformer = random_projection.GaussianRandomProjection(n_components=2)  
        mu = np.mean(x_data, axis=0)
        x_RP = transformer.fit_transform(x_data)
        t_matrix = transformer.components_
        proj = np.linalg.lstsq(x_RP.T, t_matrix)[0] + mu
        loss = ((x_data - proj) ** 2).mean() 
        print('RP loss is: ', loss)
        columns = ['class','Random component 1', 'Random component 2']
        generate_clusters('KM',x_RP, y_data,columns, 'Cluster dist. plots for each Random Projection component (K-means)', [1,2],'class',type='num')
        generate_clusters('EM',x_RP, y_data,columns, 'Cluster dist. plots for each Random Projection component (EM)', [1,2],'class',type='num')

    if tree_feature:
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(x_data, y_data)
        print(clf.feature_importances_ ) 
        model = SelectFromModel(clf, prefit=True, threshold =0.02, max_features = 2 ) # default is mean threshold
        x_FS = model.transform(x_data)
        feature_counts = x_FS.shape[1]
        print(x_FS.shape)         
        columns = ['class','Feature sel. component 1', 'Feature sel. component 2']
        generate_clusters('KM',x_FS, y_data,columns, 'Cluster dist. plots for each Feature Selection component (K-means)', [1,2],'class',type='num')
        generate_clusters('EM',x_FS, y_data,columns, 'Cluster dist. plots for each Feature Selection component (EM)', [1,2],'class',type='num')

    if tree_NN:
        num_comps = 2
        num_clusters = 3
        f1_scores, accuracys, train_times = [],[],[]
        clfs = []
        data_sets = [x_data]
        data_sets_km = [x_data]
        data_sets_em = [x_data]
        names = ['Original', 'PCA', 'ICA', 'Rand. Proj.', 'Feature Sel ']

        transformer_PCA = PCA(n_components=num_comps)
        x_PCA = transformer_PCA.fit_transform(x_data)
        data_sets.append(x_PCA)   

        clusterer = KMeans(n_clusters=num_clusters,n_init=30,max_iter=300,random_state=100)
        y_prime = clusterer.fit_predict(x_PCA)
        y_prime = np.array([y_prime]).T
        x_new = np.concatenate((x_PCA,y_prime),axis=1)
        if NN_without_org: x_new = y_prime
        data_sets_km.append(x_new)

        clusterer = GaussianMixture(n_components = num_clusters)
        y_prime = clusterer.fit_predict(x_PCA)
        y_prime = np.array([y_prime]).T
        x_new = np.concatenate((x_PCA,y_prime),axis=1)
        if NN_without_org: x_new = y_prime
        data_sets_em.append(x_new)

        transformer_ICA = FastICA(n_components=num_comps)
        x_ICA = transformer_ICA.fit_transform(x_data)  
        data_sets.append(x_ICA)    

        clusterer = KMeans(n_clusters=num_clusters,n_init=30,max_iter=300,random_state=100)
        y_prime = clusterer.fit_predict(x_ICA)
        y_prime = np.array([y_prime]).T
        x_new = np.concatenate((x_ICA,y_prime),axis=1)
        if NN_without_org: x_new = y_prime
        data_sets_km.append(x_new)

        clusterer = GaussianMixture(n_components = num_clusters)
        y_prime = clusterer.fit_predict(x_ICA)
        y_prime = np.array([y_prime]).T
        x_new = np.concatenate((x_ICA,y_prime),axis=1)
        if NN_without_org: x_new = y_prime
        data_sets_em.append(x_new)

        transformer_RP = random_projection.GaussianRandomProjection(n_components=num_comps)  
        x_RP = transformer_RP.fit_transform(x_data)
        data_sets.append(x_RP) 

        clusterer = KMeans(n_clusters=num_clusters,n_init=30,max_iter=300,random_state=100)
        y_prime = clusterer.fit_predict(x_RP)
        y_prime = np.array([y_prime]).T
        x_new = np.concatenate((x_RP,y_prime),axis=1)
        if NN_without_org: x_new = y_prime
        data_sets_km.append(x_new)

        clusterer = GaussianMixture(n_components = num_clusters)
        y_prime = clusterer.fit_predict(x_RP)
        y_prime = np.array([y_prime]).T
        x_new = np.concatenate((x_RP,y_prime),axis=1)
        if NN_without_org: x_new = y_prime
        data_sets_em.append(x_new)

        clf_FS = ExtraTreesClassifier(n_estimators=50)
        clf_FS = clf_FS.fit(x_data, y_data)
        model = SelectFromModel(clf_FS, prefit=True, threshold =0.02, max_features = 2 ) # default is mean threshold
        x_FS = model.transform(x_data)
        data_sets.append(x_FS) 

        clusterer = KMeans(n_clusters=num_clusters,n_init=30,max_iter=300,random_state=100)
        y_prime = clusterer.fit_predict(x_FS)
        y_prime = np.array([y_prime]).T
        x_new = np.concatenate((x_FS,y_prime),axis=1)
        if NN_without_org: x_new = y_prime
        data_sets_km.append(x_new)

        clusterer = GaussianMixture(n_components = num_clusters)
        y_prime = clusterer.fit_predict(x_FS)
        y_prime = np.array([y_prime]).T
        x_new = np.concatenate((x_FS,y_prime),axis=1)
        if NN_without_org: x_new = y_prime
        data_sets_em.append(x_new)

        # Experiment with NN on projected data
        if NN_KM: 
            print('K-means cluster as a feature ...')
            data_sets = data_sets_km
            print(len(data_sets))
            suffix = ' (KM)'
        elif NN_EM: 
            data_sets = data_sets_em
            print('EM cluster as a feature ...')
            data_sets = data_sets_km
            print(len(data_sets))
            suffix = ' (EM)'
        else: 
            data_sets = data_sets
            suffix = ''

        for x_data in data_sets:
            print(x_data)
            clf = mlrose.NeuralNetwork(
                hidden_nodes = [6,6], activation = 'relu', \
                algorithm = 'gradient_descent', max_iters = 1000, \
                bias = True, is_classifier = True, learning_rate = 0.0001, \
                early_stopping = True, clip_max = 5, max_attempts = 100, \
                random_state = 30)
            curves, train_score, test_score, train_acc, test_acc, train_time, test_time = \
                return_stratified_kcv_results(clf, x_data, y_data)
            f1_scores.append(test_score)
            accuracys.append(test_acc)
            print(accuracys)
            print(f1_scores)
            train_times.append(train_time)
        
        df_plot = pd.DataFrame({'names': names, 'CV_F1_Score': f1_scores,'CV_accuracy': accuracys})
    #  df_plot = pd.wide_to_long(df_plot, i=['CV_F1_Score', 'CV_accuracy'], j='Measures')
        df_plot = pd.melt(df_plot, id_vars=['names'], value_vars=['CV_F1_Score','CV_accuracy'],\
                var_name='Measures', value_name='Score')
        fig = plt.figure(1)
        ax = fig.add_subplot(121)
        sb.barplot(x="names", y="Score", hue="Measures", data=df_plot, axes=ax)
        ax.set(xlabel='dataset',ylabel='score',title='NN on org. + proj. data' + suffix)
        plt.xticks(rotation=30)    
        ax = fig.add_subplot(122)
        ax.bar(names,train_times,align='center')
        ax.set(xlabel='dataset',ylabel='Train time (s)',title='Train time of NN on org. + proj. data' + suffix)    
        fig.tight_layout()
        plt.xticks(rotation=30)
        plt.show()


    # PREPROCESS BANK DATA
    data = pd.read_csv('bank_full.csv',sep=';')
    data.drop(['day','month'],axis=1,inplace=True)
    data['y'].replace(['no'],0,inplace=True)
    data['y'].replace(['yes'],1,inplace=True)
    # convert data to numeric where possible
    data = data.apply(pd.to_numeric, errors='ignore', downcast='float')
#  print(data.hist)
    x_data = data.loc[:, data.columns != "y"]
    x_data_org = x_data
    y_data = data.loc[:, "y"]
    numerical_features = x_data.dtypes == 'float32'
    categorical_features = ~numerical_features
    columns = list(data.columns.values)
    random_state = 100
    preprocess = make_column_transformer(
        (OneHotEncoder(),categorical_features), 
        (Normalizer(), numerical_features),
        remainder="passthrough")
    x_data = preprocess.fit_transform(x_data)
    # Hold out test set for final performance measure
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=random_state, shuffle=True, stratify=y_data)

    if bank_k:
        plot_silhouette_test(x_data,'Silhouette score for Bank Marketing dataset')
        plot_sse_test(x_data,'Sum Squared Errors (K-means) for Bank Marketing dataset')
        plot_bic_test(x_data,'BIC score (Expectation Maximization) for Bank Marketing dataset')  


    if bank_cluster:
        # PLOT CLUSTER FOR K-MEANS 
        generate_clusters('KM',x_data, y_data,columns,'K-means cluster scatter plots for each attribute',[2,7], 'y', x_data_org=x_data_org)
        # PLOT CLUSTER FOR EM
        generate_clusters('EM',x_data,y_data,columns, 'EM cluster scatter plots for each attribute',[2,7], 'y',x_data_org=x_data_org)
        # PLOT CLUSTER FOR GROUND TRUTH
        plot_clusters(data,[2,7],'y', 'Ground truth cluster scatter plots for each attribute')


    if bank_pca:
        print(x_data.shape)
        transformer = PCA(n_components=8)
        x_pca = transformer.fit_transform(x_data)
        eigen_vals = transformer.explained_variance_

        proj = transformer.inverse_transform(x_pca)
        loss = ((x_data - proj) ** 2).mean()
        print('PCA loss is: ', loss)

        total_eigen = sum(eigen_vals)
        var_exp = [(i/total_eigen) for i in sorted(eigen_vals,reverse=True)]
        cum_var_exp = np.cumsum(var_exp)
        plt.bar(range(1,9),var_exp, align='center',label='individual explained variance')
        plt.step(range(1,9), cum_var_exp, where ='mid', label ='Cummulative explained variance',color='green')
        plt.xlabel('Principal component index')
        plt.ylabel('Explained variance ratio')
        plt.tight_layout()
        plt.show()

        # Sebastian Raschka, Vahid Mirjalili - Python Machine Learning_ Machine Learning and Deep Learning with Python, scikit-learn, and TensorFlow 2
        columns = ['class'] + ['principle component ' +  str(i) for i in range(1,9)]
        generate_clusters('KM',x_pca, y_data,columns, 'Bank Cluster dist. plots for each PCA component (K-means)', [2,4],'class',type='num')
        generate_clusters('EM',x_pca, y_data,columns, 'Bank Cluster dist. plots for each PCA component (EM)', [2,4],'class',type='num')

    if bank_ica:
        kurts = []
        comps = [i for i in range(2,12)]
        for i in comps:
            transformer = FastICA(n_components=i)
            x_ICA = transformer.fit_transform(x_data)    
            kurt = kurtosis(x_ICA).mean()
            print(kurt)
            kurts.append(kurt)

        plt.plot(comps,kurts)
        plt.xlabel('Components')
        plt.ylabel('Kurtosis')
        plt.title('Kurtosis plot for ICA (bank)')
        plt.xticks(comps)
        plt.show()
        
        transformer = FastICA(n_components=8)
        x_ICA = transformer.fit_transform(x_data)    
        
        proj = transformer.inverse_transform(x_ICA)
        loss = ((x_data - proj) ** 2).mean()
        print('ICA loss is: ', loss)
        columns = ['class'] + ['principle component ' +  str(i) for i in range(1,9)]
        generate_clusters('KM',x_ICA, y_data,columns, 'Bank Cluster dist. plots for each ICA component (K-means-bank)', [2,4],'class',type='num')
        generate_clusters('EM',x_ICA, y_data,columns, 'Bank Cluster dist. plots for each ICA component (EM-bank)', [2,4],'class',type='num')

    if bank_rp:
        losses, kurts = [], []
        comps = [i for i in range(1,12)]
        for i in comps:
            transformer = random_projection.GaussianRandomProjection(n_components=i)
            mu = np.mean(x_data, axis=0)
            x_RP = transformer.fit_transform(x_data)
            t_matrix = transformer.components_

            proj = np.linalg.lstsq(x_RP.T, t_matrix)[0] + mu
            loss = ((x_data - proj) ** 2).mean()
            kurt = kurtosis(x_RP).mean()
            kurts.append(kurt)
            losses.append(loss)
        fig = plt.figure(1)    
        ax = fig.add_subplot(121)
        ax.plot(comps,kurts)
        ax.set(xlabel='Components', ylabel='Kurtosis', title='Kurtosis plot for RP (Bank)',xticks=comps)

        ax = fig.add_subplot(122)
        ax.plot(comps,losses)
        ax.set(xlabel='Components', ylabel='Loss', title='Kurtosis plot for RP (Bank)',xticks=comps)

        losses, kurts = [], []

        comps = range(1,12)
        for i in comps:
            transformer = random_projection.GaussianRandomProjection(n_components=8)
            mu = np.mean(x_data, axis=0)
            x_RP = transformer.fit_transform(x_data)
            t_matrix = transformer.components_
            proj = np.linalg.lstsq(x_RP.T, t_matrix)[0] + mu
            loss = ((x_data - proj) ** 2).mean()
            kurt = kurtosis(x_RP).mean()
            kurts.append(kurt)
            losses.append(loss)
        fig = plt.figure(2)    
        ax = fig.add_subplot(121)
        ax.plot(comps,kurts)
        ax.set(xlabel='Run index', ylabel='Kurtosis', title='Kurtosis plot for RP (Bank)',xticks=comps)
        ax = fig.add_subplot(122)
        ax.plot(comps,losses)
        ax.set(xlabel='Run index', ylabel='Loss', title='Loss plot for RP (Bank)',xticks=comps)

        plt.show()

        transformer = random_projection.GaussianRandomProjection(n_components=8)  
        mu = np.mean(x_data, axis=0)
        x_RP = transformer.fit_transform(x_data)
        t_matrix = transformer.components_
        proj = np.linalg.lstsq(x_RP.T, t_matrix)[0] + mu
        loss = ((x_data - proj) ** 2).mean() 
        print('RP loss is: ', loss)
        columns = ['class'] + ['Random component ' +  str(i) for i in range(1,9)]
        generate_clusters('KM',x_RP, y_data,columns, 'Dist. plots for each Random Projection component (K-means)', [2,4],'class',type='num')
        generate_clusters('EM',x_RP, y_data,columns, 'Dist.  plots for each Random Projection component (EM)', [2,4],'class',type='num')

    if bank_feature:
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(x_data, y_data)
        print(clf.feature_importances_ ) 
        model = SelectFromModel(clf, prefit=True, threshold =0.00525,  max_features = 8 ) # default is mean threshold
        x_FS = model.transform(x_data)
        print(x_FS.shape)         
        columns = ['class'] + ['Feature selection component ' +  str(i) for i in range(1,9)]
        generate_clusters('KM',x_FS, y_data,columns, 'Cluster dist. plots for each Feature Selection component (K-means)', [2,4],'class',type='num')
        generate_clusters('EM',x_FS, y_data,columns, 'Cluster dist. plots for each Feature Selection component (EM)', [2,4],'class',type='num')

    if bank_NN:
        f1_scores, accuracys, train_times = [],[],[]
        clfs = []
        data_sets = [x_data]
        data_sets_km = [x_data]
        data_sets_em = [x_data]
        names = ['Original', 'PCA', 'ICA', 'Rand. Proj.', 'Feature Sel ']

        transformer = PCA(n_components=8)
        x_PCA = transformer.fit_transform(x_data)
        data_sets.append(x_PCA)   

        clusterer = KMeans(n_clusters=2,n_init=30,max_iter=300,random_state=100)
        y_prime = clusterer.fit_predict(x_PCA)
        y_prime = np.array([y_prime]).T
        x_new = np.concatenate((x_PCA,y_prime),axis=1)
        data_sets_km.append(x_new)

        clusterer = GaussianMixture(n_components = 2)
        y_prime = clusterer.fit_predict(x_PCA)
        y_prime = np.array([y_prime]).T
        x_new = np.concatenate((x_PCA,y_prime),axis=1)
        data_sets_em.append(x_new)

        transformer = FastICA(n_components=8)
        x_ICA = transformer.fit_transform(x_data)  
        data_sets.append(x_ICA)    

        clusterer = KMeans(n_clusters=2,n_init=30,max_iter=300,random_state=100)
        y_prime = clusterer.fit_predict(x_ICA)
        y_prime = np.array([y_prime]).T
        x_new = np.concatenate((x_ICA,y_prime),axis=1)
        data_sets_km.append(x_new)

        clusterer = GaussianMixture(n_components = 2)
        y_prime = clusterer.fit_predict(x_ICA)
        y_prime = np.array([y_prime]).T
        x_new = np.concatenate((x_ICA,y_prime),axis=1)
        data_sets_em.append(x_new)

        transformer = random_projection.GaussianRandomProjection(n_components=8)  
        x_RP = transformer.fit_transform(x_data)
        data_sets.append(x_RP) 

        clusterer = KMeans(n_clusters=2,n_init=30,max_iter=300,random_state=100)
        y_prime = clusterer.fit_predict(x_RP)
        y_prime = np.array([y_prime]).T
        x_new = np.concatenate((x_RP,y_prime),axis=1)
        data_sets_km.append(x_new)

        clusterer = GaussianMixture(n_components = 2)
        y_prime = clusterer.fit_predict(x_RP)
        y_prime = np.array([y_prime]).T
        x_new = np.concatenate((x_RP,y_prime),axis=1)
        data_sets_em.append(x_new)

        clf_FS = ExtraTreesClassifier(n_estimators=50)
        clf_FS = clf_FS.fit(x_data, y_data)
        model = SelectFromModel(clf_FS, prefit=True, threshold =0.0002, max_features = 8 ) # default is mean threshold
        x_FS = model.transform(x_data)
        data_sets.append(x_FS) 

        clusterer = KMeans(n_clusters=2,n_init=30,max_iter=300,random_state=100)
        y_prime = clusterer.fit_predict(x_FS)
        y_prime = np.array([y_prime]).T
        x_new = np.concatenate((x_FS,y_prime),axis=1)
        data_sets_km.append(x_new)

        clusterer = GaussianMixture(n_components = 2)
        y_prime = clusterer.fit_predict(x_FS)
        y_prime = np.array([y_prime]).T
        x_new = np.concatenate((x_FS,y_prime),axis=1)
        data_sets_em.append(x_new)

        # Experiment with NN on projected data
        if NN_KM: 
            print('K-means cluster as a feature ...')
            data_sets = data_sets_km
            print(len(data_sets))
            suffix = '-with K-means cluster'
        elif NN_EM: 
            data_sets = data_sets_em
            print('K-means cluster as a feature ...')
            data_sets = data_sets_km
            print(len(data_sets))
            suffix = '-with EM cluster'
        else: 
            data_sets = data_sets
            suffix = ''

        for x_data in data_sets:
            print(x_data.shape)
            clf = mlrose.NeuralNetwork(
                hidden_nodes = [6,6], activation = 'relu', \
                algorithm = 'gradient_descent', max_iters = 1000, \
                bias = True, is_classifier = True, learning_rate = 0.0001, \
                early_stopping = True, clip_max = 5, max_attempts = 100, \
                random_state = 30)
            curves, train_score, test_score, train_acc, test_acc, train_time, test_time = \
                return_stratified_kcv_results(clf, x_data, y_data)
            f1_scores.append(test_score)
            accuracys.append(test_acc)
            print(accuracys)
            print(f1_scores)
            train_times.append(train_time)
        
        df_plot = pd.DataFrame({'names': names, 'CV_F1_Score': f1_scores,'CV_accuracy': accuracys})
    #  df_plot = pd.wide_to_long(df_plot, i=['CV_F1_Score', 'CV_accuracy'], j='Measures')
        df_plot = pd.melt(df_plot, id_vars=['names'], value_vars=['CV_F1_Score','CV_accuracy'],\
                var_name='Measures', value_name='Score')
        fig = plt.figure(1)
        ax = fig.add_subplot(121)
        sb.barplot(x="names", y="Score", hue="Measures", data=df_plot, axes=ax)
        ax.set(xlabel='dataset',ylabel='score',title='NN on original + proj. data' + suffix)
        plt.xticks(rotation=30)    
        ax = fig.add_subplot(122)
        ax.bar(names,train_times,align='center')
        ax.set(xlabel='dataset',ylabel='Train time (s)',title='NN on original + proj. data' + suffix)    
        fig.tight_layout()
        plt.xticks(rotation=30)
        plt.show()

if __name__ == "__main__" :
    import argparse
    print("Running Unsupervised Learning ...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='choose_k')
    parser.add_argument('--dataset', default='tree')


    args = parser.parse_args()
    task = args.task
    dataset = args.dataset
    if dataset == 'tree':
        if task == 'choose_k':
            print("Finding best k for tree:...")
            main(tree_k=1)
        if task== 'cluster':
            print("Clustering tree data:...")
            main(tree_cluster=1)
        if task== 'pca':
            print("Run tree PCA:...")
            main(tree_pca=1)
        if task== 'ica':
            print("Run tree ICA:...")
            main(tree_ica=1)
        if task== 'rp':
            print("Run tree RP:...")
            main(tree_rp=1)
        if task== 'feature':
            print("Run tree feature selection:...")
            main(tree_feature=1)
        if task== 'NN':
            print("Run NN experiements on the Tree Dataset...")
            main(tree_NN=1)
        if task== 'NN_clustering_KM':
            print("Run NN with clustering experiements (KM) on the Tree Dataset...")
            main(tree_NN=1, NN_KM=1)
        if task== 'NN_clustering_EM':
            print("Run NN with clustering experiements (EM) on the Tree Dataset...")
            main(tree_NN=1, NN_EM=1)
        if task== 'NN_clustering_KM_without_org':
            print("Run NN with clustering experiements on the Tree Dataset...")
            main(tree_NN=1, NN_KM=1, NN_without_org=1)
        if task== 'NN_clustering_EM_without_org':
            print("Run NN with clustering experiements on the Tree Dataset...")
            main(tree_NN=1, NN_EM=1, NN_without_org=1)

    if dataset == 'bank':
        if task == 'choose_k':
            print("Finding best k for bank:...")
            main(bank_k=1)
        if task== 'cluster':
            print("Clustering bank data:...")
            main(bank_cluster=1)
        if task== 'pca':
            print("Run bank PCA:...")
            main(bank_pca=1)
        if task== 'ica':
            print("Run bank ICA:...")
            main(bank_ica=1)
        if task== 'rp':
            print("Run bank RP:...")
            main(bank_rp=1)
        if task== 'feature':
            print("Run bank feature selection:...")
            main(bank_feature=1)
        if task== 'NN':
            print("There is no NN experiements on the Bank Dataset")

#____________________________________
# REF CODE
'''
  # sb.violinplot(y="Mean_Green",hue = 'class', data = df)

   
  #  sb.stripplot(x = "class", y = "Mean_Green", data = df, jitter = False)


    plt.figure(1)
    df.sort_values(by='class',ascending=False)  
    pc = parallel_coordinates(df, 'class', color=[[1,0,0,0.01],[0,1,0,0.6]])


    plt.figure(2)
    y_data = np.array([y_data]).transpose()
    x_data_cluster = np.concatenate((y_data, x_data),axis=1)
    df = pd.DataFrame(data=x_data_cluster,columns = columns) 

    df.sort_values(by='class',ascending=False)   
#    pc = parallel_coordinates(df, 'class', color=('#FFE888', '#FF9999'))

  #  pc = parallel_coordinates(df, 'class', color=('#FFE888'),alpha=1)
  #  pc = parallel_coordinates(df, 'class', color=('#FFE888','#FF9999'),alpha=0.1) 
    pc = parallel_coordinates(df, 'class', color=[[1,0,0,0.01],[0,1,0,0.9]])
'''

