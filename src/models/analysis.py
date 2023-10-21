import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.offline as pyo
import seaborn as sns
from src.input_output_functions import append_path

sns.set_theme(style='darkgrid', palette='husl')
pyo.init_notebook_mode()


class GPSVisualizer:

    def __init__(self, df, gps_df):
        self.df = pd.merge(df, gps_df, on='unique_id', how='left')
        # self.df['inverse_pop'] = 1 / self.df['est_total_pop']

    def visualize(self,
                  color='cluster_label',
                  size='est_total_pop',
                  hover_name='school',
                  hover_data=['school_grade', 'est_total_pop']):

        self.df['cluster_label'] = self.df['cluster_label'].astype('string')

        fig = px.scatter_mapbox(
            self.df,
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
            color_discrete_sequence=px.colors.qualitative.Vivid
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
                'sizemin': 8,
            }
        )
        fig.show()


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

