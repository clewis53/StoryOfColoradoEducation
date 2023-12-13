from gower import gower_matrix
from kmodes.kprototypes import KPrototypes
import lightgbm as lgb
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.offline as pyo
import prince
import seaborn as sns
import shap
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import Normalizer
from yellowbrick.cluster import KElbowVisualizer

import src.features.preprocessors as preprocessors
from src.input_output_functions import append_path

sns.set_theme(style='darkgrid', palette='husl')
pyo.init_notebook_mode()

RANDOM_STATE = 42


class DBSCANSelector:
    """ Class that has a collection of methods that help describe the model for different values of eps """

    def __init__(self, df):
        # The features
        self.X = df.drop(['unique_id', 'year'], axis=1)
        # the index part of the dataframe
        self.index = df[['unique_id', 'year']]

    def show_elbow(self):
        n = 2 * self.X.shape[1] - 1

        neighbors = NearestNeighbors(n_neighbors=n, radius=1).fit(self.X)
        distances, indices = neighbors.kneighbors(self.X)
        distances = np.sort(distances, axis=0)[:, n-1]

        plt.figure(dpi=100)
        sns.lineplot(distances)
        plt.xlabel('Points in Dataset')
        plt.ylabel('epsilon')
        plt.show()


class KMeansSelector:
    """ Class that has a collection of methods that help describe the model for different values of K """
    K_MIN = 2
    K_MAX = 13

    def __init__(self, df):
        # The features
        self.X = df.drop(['unique_id', 'year'], axis=1)
        # the index part of the dataframe
        self.index = df[['unique_id', 'year']]

    @staticmethod
    def create_model(n_clusters=8):
        """ Returns a KMeans model """
        return KMeans(n_clusters=n_clusters, init='k-means++', random_state=RANDOM_STATE, n_init='auto', max_iter=1000)

    def show_elbow(self, save_filename=None):
        """ Utilizes the KElbowVisualizer from yellowbrick to create an elbow plot
        of distortion scores and model fit time across a range of k values and identifies
         the optimal k"""
        # Ensure that school_id and year are not found in the dataframe

        model = self.create_model()
        visualizer = KElbowVisualizer(model, k=(self.K_MIN, self.K_MAX))

        visualizer.fit(self.X)
        visualizer.show(outpath=save_filename)

    def show_silhouettes(self, k_min=None, k_max=None):
        """ Shows silhouette plots across a range from k_min to k_max """
        if k_min is None:
            k_min = self.K_MIN
        if k_max is None:
            k_max = self.K_MAX

        if k_min < 2 or k_min > k_max:
            raise ValueError('k_min must be greater than 1 and less than k_max')

        n_cols = 2
        n_rows = math.ceil(float(k_max - k_min) / 2)

        for k in range(k_min, k_max):
            self.make_silhouette_plot(k)

    def get_silhouette_scores(self, n_clusters):
        """ Returns the silhouette score and sample silhouette scores for the specified number of clusters. """
        model = self.create_model(n_clusters)
        labels = model.fit_predict(self.X)

        # Find the average Silhouette Score
        sil_avg = silhouette_score(self.X, labels)

        # Find Silhouette Scores for each sample and plot them
        sample_sil_values = silhouette_samples(self.X, labels)

        return sil_avg, sample_sil_values, labels

    def make_silhouette_plot(self, n_clusters):
        """ Makes a silhouette plot for the specified number of clusters """
        sil_avg, sample_sil_values, labels = self.get_silhouette_scores(n_clusters)

        print(f'For {n_clusters=}',
              f'The average silhouette score is {sil_avg}')

        # Initialize plot
        plt.title(f'Silhouette Plot For {n_clusters=}')
        plt.xlabel('Silhouette Coefficient Values')
        plt.ylabel('Cluster Label')
        plt.axvline(x=sil_avg, color='k', linestyle='-.')
        plt.xlim([-0.1, 1])
        plt.ylim([0, len(self.X) + (n_clusters + 1) * 10])
        plt.yticks([])
        # Initialize y_lower for plotting cluster silhouettes
        y_lower = 10
        colors = sns.husl_palette(n_clusters)
        for i in range(n_clusters):
            # Find silhouette scores for ith cluster and sort
            cluster_sil_values = sample_sil_values[labels == i]
            cluster_sil_values.sort()
            # Plot silhouette
            size = cluster_sil_values.shape[0]
            y_upper = y_lower + size  # Set the top of the silhouette
            sns.husl_palette()
            color = colors[i]  # Create a color for the silhouette
            plt.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_sil_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7
            )
            # Write the cluster label in the middle
            plt.text(-0.05, y_lower + 0.5 * size, f'{i}')
            # Update y_lower
            y_lower = y_upper + 10
        plt.show()


class KPrototypesModelSelector(KMeansSelector):
    def __init__(self, df, high_school=False):
        kpromodel = KPrototypesModel(df, 2, high_school=high_school)
        self.X = kpromodel.X
        self.index = kpromodel.index
        self.cat_indices = kpromodel.cat_indices

    @staticmethod
    def create_model(n_clusters=8):
        """ Returns a KMeans model """
        return KPrototypes(n_clusters=n_clusters, random_state=RANDOM_STATE)

    def show_elbow(self, save_filename=None):
        """ Utilizes the KElbowVisualizer from yellowbrick to create an elbow plot
        of distortion scores and model fit time across a range of k values and identifies
         the optimal k"""
        # Ensure that school_id and year are not found in the dataframe
        costs = np.zeros(self.K_MAX-self.K_MIN)
        ks = np.arange(self.K_MIN, self.K_MAX)
        for i in range(len(ks)):
            model = self.create_model(ks[i])
            model.fit_predict(self.X, categorical=self.cat_indices)
            costs[i] = model.cost_

        plt.plot(ks, costs, linestyle='--', marker='o')
        plt.title('KPrototypes Elbow')
        plt.xlabel('K')
        plt.ylabel('Cost')
        if save_filename is not None:
            plt.savefig(save_filename)
        plt.show()

    def get_silhouette_scores(self, n_clusters):
        """ Returns the silhouette score and sample silhouette scores for the specified number of clusters. """
        model = self.create_model(n_clusters)
        labels = model.fit_predict(self.X, categorical=self.cat_indices)

        distance_mat = gower_matrix(self.X)
        # Find the average Silhouette Score
        sil_avg = silhouette_score(distance_mat, labels, metric='precomputed')

        # Find Silhouette Scores for each sample and plot them
        sample_sil_values = silhouette_samples(distance_mat, labels, metric='precomputed')

        return sil_avg, sample_sil_values, labels


class KMeansModel:
    """ Class that will create KMeans model and provide evaluations of that model. """

    def __init__(self, df, n_clusters=4):
        # The index columns
        self.index = df[['unique_id', 'year']]
        # The feature columns
        self.X = df.drop(['unique_id', 'year'], axis=1)
        # The model that will be used
        self.model = KMeans(n_clusters, random_state=RANDOM_STATE, max_iter=100, n_init='auto')
        # Initialize the labels assigned
        self.labels = []

    def fit_predict(self):
        """ Fits and predicts the model """
        self.labels = self.model.fit_predict(self.X)

    def save_model(self, output_filepath, filename, data_no_outliers=None):
        """ Saves a dataframe composed of the input features and assigned cluster labels. """
        if data_no_outliers is None:
            data_no_outliers = pd.concat((self.index, self.X), axis=1)

        labels = pd.Series(self.labels, name='cluster_label')
        df = pd.concat((data_no_outliers, labels), axis=1)
        df.to_csv(append_path(output_filepath, filename), index=False)

    def evaluate_model(self):
        """ Shows Davies Bouldin Index (close to zero represents a good model)
        the Calinski Harabaz Index (the higher the better)
        and the Silhouette Score (where 1 is the ideal value) """
        dbs = davies_bouldin_score(self.X, self.labels)
        cs = calinski_harabasz_score(self.X, self.labels)
        ss = silhouette_score(self.X, self.labels)
        print(f'Davies Bouldin Score: {dbs}',
              f'Calinksi Score: {cs}',
              f'Silhouette Score: {ss}', sep='\n')

    def show_feature_importance(self, data_no_outliers=None, save_filename=None):
        """ Uses a gradient boosting decision tree from LightGBM to classify labels,
        and then shows the feature importance using the shap Tree explainer """
        # If not no outlier dataset is provided, the data used for the clustering model will be used
        if data_no_outliers is None:
            data_no_outliers = self.X

        # Using LGBM classify clusters
        lgbm = lgb.LGBMClassifier(colsample_bytree=0.8, verbose=-1)
        lgbm.fit(X=data_no_outliers, y=self.labels)

        # Create an Explainer
        explainer = shap.TreeExplainer(lgbm)
        shap_vals = explainer.shap_values(data_no_outliers)

        # Display the Explainer
        shap.summary_plot(shap_vals, data_no_outliers, plot_type='bar', plot_size=(15, 10), show=False)
        plt.title('SHAP Summary Plot of LGBM Cluster Classification')
        if save_filename is not None:
            plt.savefig(save_filename)
        plt.show()

    def _get_pca(self, n_dim):
        """ Get the PCA decomposition for the provided number of components """
        pca = prince.PCA(
            n_components=n_dim,
            n_iter=3,
            rescale_with_mean=True,
            rescale_with_std=True,
            copy=True,
            check_input=True,
            engine='sklearn',
            random_state=RANDOM_STATE
        )
        pca.fit(self.X)
        df = pca.transform(self.X)
        df.columns = [f'comp{i}' for i in range(n_dim)]

        return pca, df

    def plot_2d(self, save_filename=None):
        """ Plot the dataframe in a 2 dimensional space """
        # Break down the dataframe to two principal components
        pca, df_2d = self._get_pca(2)

        # Display the explained variance for those principal components
        explained_var = pca.eigenvalues_summary
        print(f'Explained Variance for Model {explained_var}')
        var_percents = pca.percentage_of_variance_

        # Make a plot of the dataframe colored by cluster
        sns.scatterplot(data=df_2d, x='comp0', y='comp1', hue=self.labels)
        plt.title('PCA 2D Plot')
        plt.xlabel(f'Comp 1 -- {var_percents[0]/100:.2%}')
        plt.ylabel(f'Comp 2 -- {var_percents[1]/100:.2%}')

        if save_filename is not None:
            plt.savefig(save_filename)

        plt.show()

    def plot_3d(self, high_school=False):
        """ Plot the dataframe in a 3 dimensional space """
        # Break down the dataframe into 3D space
        pca, df_3d = self._get_pca(3)

        # Display the explained variance
        explained_var = pca.eigenvalues_summary
        print(f'Explained Variance for Model {explained_var}')

        # Make the plot
        if high_school:
            size, opacity = 5, 0.9
        else:
            size, opacity = 2, 0.75

        fig = px.scatter_3d(
            df_3d,
            x='comp0',
            y='comp1',
            z='comp2',
            color=self.labels,
            template='plotly',
            title='PCA 3D',
            color_discrete_sequence=px.colors.qualitative.Vivid
        ).update_traces(
            # mode = 'markers',
            marker={
                "size": size,
                "opacity": opacity,
                "line": {
                    "width": 0.1,
                    "color": "black",
                }
            }
        ).update_layout(
            width=800,
            height=800,
            autosize=True,
            showlegend=True,
        )

        fig.show()


class KPrototypesModel(KMeansModel):

    def __init__(self, df, n_clusters, high_school=False, **kwargs):
        super().__init__(df, n_clusters)

        self.num_cols = preprocessors.NUM_COLS
        if high_school:
            self.num_cols = self.num_cols + preprocessors.HIGH_NUM_COLS

        # The features
        self.X = df.drop(['unique_id', 'year'], axis=1)
        # the index part of the dataframe
        self.index = df[['unique_id', 'year']]
        # The location of the categorical columns
        non_cat_indices = {self.X.columns.get_loc(col) for col in self.num_cols}
        self.cat_indices = list(set(np.arange(len(self.X.columns))).difference(non_cat_indices))
        # Normalize numeric columns
        self.transform_df()

        self.model = KPrototypes(n_clusters=n_clusters, random_state=RANDOM_STATE)

    def fit_predict(self):
        self.labels = self.model.fit_predict(self.X, categorical=self.cat_indices)

    def transform_df(self):
        # Normalize the numeric columns
        normalizer = Normalizer()
        self.X[self.num_cols] = normalizer.fit_transform(self.X[self.num_cols])


class DBSCANModel(KMeansModel):

    def __init__(self, df, eps, min_samples, **kwargs):
        # The index columns
        self.index = df[['unique_id', 'year']]
        # The feature columns
        self.X = df.drop(['unique_id', 'year'], axis=1)
        # The model that will be used
        self.model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        # Initialize the labels assigned
        self.labels = []


def show_adj_rand_score_mat(models, model_names):
    """ Calculates the adjusted rand score between each models labels and returns it as a pandas DataFrame """
    if len(models) != len(model_names):
        raise ValueError(f"""The number of models provided was {len(models)}\n
        while the number of names was {len(model_names)}. 
        These must be the same""")

    mat = np.eye(len(models))
    for i in range(len(models)):
        for j in range(i):
            mat[i, j] = adjusted_rand_score(models[i].labels, models[j].labels)

    return pd.DataFrame(mat, columns=model_names, index=model_names)


