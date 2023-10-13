import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from pyod.models.ecod import ECOD


# from sentence_transformers import SentenceTransformer


def main(input_filepath, output_filepath):
    all_interim_df = pd.read_csv(input_filepath.joinpath('all_data.csv'))
    high_interim_df = pd.read_csv(input_filepath.joinpath('high_school.csv'))

    preprocessor = Preprocessor()
    processed_df = preprocessor.fit_transform(all_interim_df)

    # num = Pipeline(steps=[
    #     ('encoder', Normalizer())
    # ])

    # feature_builder = ColumnTransformer(
    #     transformers=[
    #         ('num', num, all_num_cols)
    #     ], remainder='passthrough', verbose_feature_names_out=False)
    #
    # preprocessor = Pipeline(
    #     steps=[
    #         ('NA_filler', FillBackForward()),
    #         ('preprocessor', feature_builder),
    #         ('outlier_remover', OutlierRemover(cols=all_cols))])
    # pipe_fit = preprocessor.fit(all_interim_df[all_cols + index_cols])
    # processed_df = pd.DataFrame(pipe_fit.transform(all_interim_df[all_cols + index_cols]),
    #                             columns=pipe_fit[1].get_feature_names_out())

    processed_df.to_csv(output_filepath.joinpath('all_data.csv'), index=False)

    # feature_builder = ColumnTransformer(
    #     transformers=[
    #         ('num', num, all_num_cols + high_num_cols)
    #     ], remainder='passthrough', verbose_feature_names_out=False)
    #
    # preprocessor = Pipeline(
    #     steps=[
    #         ('NA_filler', FillBackForward()),
    #         ('feature_builder', feature_builder),
    #         ('outlier_remover', OutlierRemover(cols=(all_cols + high_cols)))])
    # pipe_fit = preprocessor.fit(high_interim_df[all_cols + high_cols + index_cols])
    # processed_df = pd.DataFrame(pipe_fit.transform(high_interim_df[all_cols + high_cols + index_cols]),
    #                             columns=pipe_fit[1].get_feature_names_out())
    # processed_df.to_csv(output_filepath.joinpath('high_school.csv'), index=False)


class Preprocessor(TransformerMixin, BaseEstimator):
    INDEX_COLS = ['school_id', 'year']

    CAT_COLS = ['pct_amind',
                'pct_asian',
                'pct_black',
                'pct_hisp',
                'pct_white',
                'pct_2ormore',
                'pct_fr',
                'achievement_dir',
                'growth_dir',
                'overall_dir',
                'school_grade']

    NUM_COLS = ['est_total_pop',
                'child_pov_ratio',
                'child_adult_ratio',
                'instruction_per_pupil',
                'support_per_pupil',
                'community_per_pupil',
                'other_per_pupil',
                'pct_amind',
                'pct_asian',
                'pct_black',
                'pct_hisp',
                'pct_white',
                'pct_2ormore',
                'pct_fr', ]

    HIGH_CAT_COLS = ['eng_yn',
                     'math_yn',
                     'read_yn',
                     'sci_yn']

    HIGH_NUM_COLS = ['pct_remediation',
                     'graduation_rate']

    def __init__(self, index_cols=None, num_cols=None, remainder_cols=None, high_school=False, **kwargs):
        if index_cols is None:
            index_cols = self.INDEX_COLS

        if num_cols is None:
            if not high_school:
                num_cols = self.NUM_COLS
            else:
                num_cols = self.NUM_COLS + self.HIGH_NUM_COLS

        if remainder_cols is None:
            if not high_school:
                remainder_cols = self.INDEX_COLS + self.CAT_COLS
            else:
                remainder_cols = self.INDEX_COLS + self.CAT_COLS + self.HIGH_CAT_COLS

        na_filler = FillBackForward()
        feature_builder = FeatureBuilder(num_cols=num_cols, remainder_cols=remainder_cols)
        outlier_remover = OutlierRemover(index_cols=index_cols)

        steps = [('NA_filler', na_filler),
                 ('feature_builder', feature_builder),
                 'outlier_remover', outlier_remover]

        self.pipeline = Pipeline(steps=steps)

    def fit(self, X, y=None):
        self.pipeline.fit(X)

    def transform(self, X):
        return self.pipeline.transform(X)


class FeatureBuilder(TransformerMixin, BaseEstimator):
    VERBOSE_FEATURE_NAMES_OUT = False

    def __init__(self, num_cols=None, remainder_cols=None, **kwargs):
        num = Pipeline(steps=[('encoder', Normalizer())])
        remainder = Pipeline(steps=[('encoder', PassthroughTransformer())])
        self.transformer = ColumnTransformer(transformers=[
            ('num', num, num_cols),
            ('', remainder, remainder_cols)],
            verbose_feature_names_out=self.VERBOSE_FEATURE_NAMES_OUT)

    def fit(self, X, y=None):
        self.transformer.fit(X)

    def transform(self, X):
        return self.transformer.transform(X)

class PassthroughTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, **kwargs):
        return X


class FillBackForward(BaseEstimator, TransformerMixin):
    NO_FILL = ['school', 'school_id', 'district_name', 'district_id']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X[X['school_id'].notna()]
        X = self.fill_back_forward(X)
        X = self.fill_median_by_year(X)

        return X

    @staticmethod
    def fill_back_forward(X):
        """ Fills NA values for each school with values from the most recent year first.
        Then fills NA values from the previous year."""
        ids = X['school_id'].unique()
        for i in ids:
            X.loc[X['school_id'] == i] = X.loc[X['school_id'] == i].sort_values(by='year').bfill()
            X.loc[X['school_id'] == i] = X.loc[X['school_id'] == i].sort_values(by='year').ffill()

        return X

    @staticmethod
    def fill_median_by_year(X):
        """ Fills NA values with the median of each year."""
        years = X['year'].unique()
        for year in years:
            X.loc[X['year'] == year] = X.loc[X['year'] == year].fillna(X.loc[X['year'] == year].median())

        return X


class OutlierRemover(BaseEstimator, TransformerMixin):

    def __init__(self, index_cols=None, **kwargs):
        if index_cols is None:
            self.end_index_cols = 2
        else:
            self.end_index_cols = len(index_cols)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        clf = ECOD()
        clf.fit(X[:, self.end_index_cols:])
        outliers = clf.predict(X[:, self.end_index_cols:])

        X = X[outliers == 0]

        return X


class LLC(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, cols):
        sentences = X.apply(lambda x: self.compile_text(x), axis=1).tolist()

        model = SentenceTransformer(r"sentence-transformers/paraphrase-MiniLM-L6-v2")
        output = model.encode(sentences=sentences,
                              show_progress_bar=True,
                              normalize_embeddings=True)

        df_embedding = pd.DataFrame(output)
        return df_embedding

    @staticmethod
    def compile_text(X, cols):
        return ',\n'.join([f'{col}: {X[col]}' for col in cols])


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[2]
    input_filepath = project_dir.joinpath("data/interim")
    output_filepath = project_dir.joinpath("data/processed")

    main(input_filepath, output_filepath)
