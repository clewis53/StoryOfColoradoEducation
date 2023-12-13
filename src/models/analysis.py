import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.offline as pyo
import seaborn as sns
from src.input_output_functions import append_path
from src.features.preprocessors import CAT_COLS, HIGH_CAT_COLS

sns.set_theme(style='darkgrid', palette='husl')
pyo.init_notebook_mode()


class GPSVisualizer:

    def __init__(self, df, gps_df):
        self.df = pd.merge(df, gps_df, on='unique_id', how='left')
        cluster_labels = self.df['cluster_label'].unique()
        cluster_labels.sort()
        color_seq = px.colors.qualitative.Vivid
        self.color_map = {str(cluster_labels[i]): color_seq[i] for i in range(len(cluster_labels))}

    def visualize(
            self,
            color='cluster_label',
            size='school_grade',
            hover_name='school',
            hover_data=['school_grade', 'total'],
            cluster_label=None,
            year=None
    ):
        if type(year) == 'int':
            x = self.filter_df(cluster_label=cluster_label, year=year)
            x['cluster_label'] = x['cluster_label'].astype('string')
        elif year is None:
            x = self.jitter_loc()

        cluster_labels = self.df['cluster_label'].unique

        fig = px.scatter_mapbox(
            x,
            lat='latitude',
            lon='longitude',
            hover_name=hover_name,
            hover_data=hover_data,
            color=color,
            text=color,
            size=size,
            size_max=20,
            zoom=7,
            height=800,
            width=1600,
            color_discrete_map=self.color_map,
            animation_frame='year'
        ).update_layout(
            mapbox_style='carto-positron',
            margin={
                'r': 0,
                't': 0,
                'l': 0,
                'b': 0
            }
        ).update_traces(
            marker={
                'opacity': 0.33,
                # 'sizemin': 4,
            }
        )
        fig.show()

    def filter_df(self, **kwargs):
        x = self.df.copy()
        for key, value in kwargs.items():
            if value is not None:
                x = x.loc[x[key] == value]
        return x

    def jitter_loc(self):
        x = self.df.copy()
        x['cluster_label'] = x['cluster_label'].astype('string')

        scale = 0.0005
        x['latitude'] = x['latitude'] + np.random.normal(0, scale, len(x['latitude']))
        x['longitude'] = x['longitude'] + np.random.normal(0, scale, len(x['longitude']))

        return x


class ClusterChanges:

    def __init__(self, df):
        self.change_df = pd.pivot(data=df, index='unique_id', columns='year', values='cluster_label')
        self.original_df = df

    def count_changes(self, years=(2010, 2011, 2012)):
        changes = 0
        for i in range(len(years) - 1):
            df = self.change_df[[years[i], years[i+1]]].dropna()
            changes += (df[years[i]] != df[years[i+1]]).sum()
        return changes

    def pct_changes(self, years=(2010, 2011, 2012)):
        changes = self.count_changes(years)
        possible_changes = len(self.change_df) * (len(years) - 1)
        return changes / possible_changes

    def schools_that_changed(self, first_year, second_year):
        df = self.change_df[[first_year, second_year]].dropna()
        change_loc = df[first_year] != df[second_year]
        ids = df.index[change_loc]

        original_id_loc = self.original_df['unique_id'].isin(ids)
        year_loc = (self.original_df['year'] == first_year) | (self.original_df['year'] == second_year)
        return pd.pivot(self.original_df[(original_id_loc) & (year_loc)], index='unique_id', columns='year')


class PairPlot:
    cat_cols = [
        'overall_dir',
        'achievement_dir',
        'growth_dir',
        'school_grade',

    ]

    def __init__(self, df, cols):
        self.df = df.copy()[cols]
        self.jitter()

    def visualize(self):
        n_colors = len(self.df['cluster_label'].unique())
        g = sns.PairGrid(self.df, hue='cluster_label', diag_sharey=False, corner=True, palette=sns.color_palette("Paired")[:n_colors])
        g.map_diag(sns.histplot, multiple='stack')
        g.map_upper(sns.scatterplot, alpha=0.5)
        g.map_lower(sns.kdeplot)
        plt.show()

    def jitter(self):
        large_scale = 0.18
        small_scale = 0.01
        for col in self.df.columns:
            if col == 'cluster_label':
                continue
            if col in CAT_COLS + HIGH_CAT_COLS + ['emh_combined_True']:
                self.df[col] = self.df[col] + np.random.normal(0, large_scale, len(self.df))
            else:
                self.df[col] = self.df[col] + np.random.normal(0, small_scale, len(self.df))


class RepeatCounter:

    def __init__(self, df, schools):
        self.w_names = pd.merge(df, schools, on='unique_id', how='left')

    def count_repeats(self, cluster_label, criteria=2):
        cluster_3 = self.w_names.loc[self.w_names['cluster_label'] == cluster_label, ['school', 'year']].sort_values(
            by=['school', 'year'])
        count_3 = cluster_3.groupby('school').count()
        repeats = count_3[count_3['year'] >= criteria]
        pct_repeats = len(repeats) / len(count_3)
        print(f'The percentage of schools that stayed in cluster {cluster_label} for {criteria} or more years was {pct_repeats :.2%}')
        return repeats
