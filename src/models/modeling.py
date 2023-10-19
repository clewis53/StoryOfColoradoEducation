from kmodes.kprototypes import KPrototypes
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.offline as pyo
import prince
import seaborn as sns
import shap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from yellowbrick.cluster import KElbowVisualizer

from src.input_output_functions import append_path

sns.set_theme(style='darkgrid', palette='husl')
pyo.init_notebook_mode()

RANDOM_STATE = 42


class KMeansSelector:
    """ Class that has a collection of methods that help describe the model for different values of K """
    K_MIN = 2
    K_MAX = 13

    def __init__(self, df):
        # The features
        self.X = df.drop(['school_id', 'year'], axis=1)
        # the index part of the dataframe
        self.index = df[['school_id', 'year']]

    @staticmethod
    def create_model(n_clusters=8):
        """ Returns a KMeans model """
        return KMeans(n_clusters=n_clusters, init='k-means++', random_state=RANDOM_STATE, n_init='auto')

    def show_elbow(self):
        """ Utilizes the KElbowVisualizer from yellowbrick to create an elbow plot
        of distortion scores and model fit time across a range of k values and identifies
         the optimal k"""
        # Ensure that school_id and year are not found in the dataframe

        model = self.create_model()
        visualizer = KElbowVisualizer(model, k=(self.K_MIN, self.K_MAX))

        visualizer.fit(self.X)
        visualizer.show()

    def show_silhouettes(self, k_min=None, k_max=None):
        """ Shows silhouette plots across a range from k_min to k_max """
        if k_min is None:
            k_min = self.K_MIN
        if k_max is None:
            k_max = self.K_MAX

        if k_min < 2 or k_min > k_max:
            raise ValueError('k_min must be greater than 1 and less than k_max')

        for k in range(k_min, k_max):
            print(f'{k=}')
            self.make_silhouette_plot(k)

    def make_silhouette_plot(self, n_clusters):
        """ Makes a silhouette plot for the specified number of clusters """
        model = self.create_model(n_clusters)
        labels = model.fit_predict(self.X)

        # Find the average Silhouette Score
        sil_avg = silhouette_score(self.X, labels)
        print(f'For {n_clusters=}',
              f'The average silhouette score is {sil_avg}')

        # Find Silhouette Scores for each sample and plot them
        sample_sil_values = silhouette_samples(self.X, labels)
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


class KMeansModel:
    """ Class that will create KMeans model and provide evaluations of that model. """

    def __init__(self, df, n_clusters):
        self.index = df[['school_id', 'year']]
        self.X = df.drop(['school_id', 'year'], axis=1)
        self.model = KMeans(n_clusters, random_state=RANDOM_STATE, max_iter=100, n_init='auto')
        self.labels = []

    def fit_predict(self):
        """ Fits and predicts the model """
        self.labels = self.model.fit_predict(self.X)

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

    def show_feature_importance(self, data_no_outliers=None):
        """ Uses a gradient boosting decision tree from LightGBM to classify labels,
        and then shows the feature importance using the shap Tree explainer """
        if data_no_outliers is None:
            data_no_outliers = self.X

        lgbm = lgb.LGBMClassifier(colsample_bytree=0.8, verbose=-1)
        lgbm.fit(X=data_no_outliers, y=self.labels)

        explainer = shap.TreeExplainer(lgbm)
        shap_vals = explainer.shap_values(data_no_outliers)
        shap.summary_plot(shap_vals, data_no_outliers, plot_type='bar', plot_size=(15, 10))

    def _get_pca(self, n_dim):
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

    def plot_2d(self):
        pca, df_2d = self._get_pca(2)

        explained_var = pca.eigenvalues_summary
        print(f'Explained Variance for Model {explained_var}')
        sns.scatterplot(data=df_2d, x='comp0', y='comp1', hue=self.labels)
        plt.show()

    def plot_3d(self):
        pca, df_3d = self._get_pca(3)

        explained_var = pca.eigenvalues_summary
        print(f'Explained Variance for Model {explained_var}')

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
                "size": 2,
                "opacity": 0.6,
                # "symbol" : "diamond",
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
            legend=dict(title_font_family="Times New Roman",
                        font=dict(size=20)),
            scene=dict(xaxis=dict(title='comp1', titlefont_color='black'),
                       yaxis=dict(title='comp2', titlefont_color='black'),
                       zaxis=dict(title='comp3', titlefont_color='black')),
            font=dict(family="Gilroy", color='black', size=15))

        fig.show()


class KPrototypesModel(KMeansModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = KPrototypesModel(**kwargs)