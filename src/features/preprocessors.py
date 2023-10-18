import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from pyod.models.ecod import ECOD
from sentence_transformers import SentenceTransformer


class KMeansPreprocessor(BaseEstimator, TransformerMixin):
    INDEX_COLS = ['unique_id', 'year']

    CAT_COLS = ['achievement_dir',
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
        # Set defaults if no information was provided
        if index_cols is None:
            self.index_cols = self.INDEX_COLS
        else:
            self.index_cols = index_cols

        if num_cols is None:
            if not high_school:
                self.num_cols = self.NUM_COLS
            else:
                self.num_cols = self.NUM_COLS + self.HIGH_NUM_COLS
        else:
            self.num_cols = num_cols

        if remainder_cols is None:
            if not high_school:
                self.remainder_cols = self.INDEX_COLS + self.CAT_COLS
            else:
                self.remainder_cols = self.INDEX_COLS + self.CAT_COLS + self.HIGH_CAT_COLS
        else:
            self.remainder_cols = index_cols + remainder_cols

        na_filler = FillBackForward()
        feature_builder = KMeansFeatureBuilder(num_cols=self.num_cols, remainder_cols=self.remainder_cols)
        outlier_remover = OutlierRemover(index_cols=self.index_cols)

        steps = [('NA_filler', na_filler),
                 ('feature_builder', feature_builder),
                 ('outlier_remover', outlier_remover)]

        self.pipeline = Pipeline(steps=steps)

    def fit(self, X, y=None):
        self.pipeline.fit(X)
        return self

    def transform(self, X):
        X = self.pipeline.transform(X)
        return pd.DataFrame(X, columns=(self.remainder_cols + self.num_cols))


class KMeansFeatureBuilder(BaseEstimator, TransformerMixin):
    VERBOSE_FEATURE_NAMES_OUT = False

    def __init__(self, num_cols=None, remainder_cols=None):
        num = Pipeline(steps=[('encoder', Normalizer())])
        passthrough = Pipeline(steps=[('encoder', PassthroughTransformer())])
        transformer = ColumnTransformer(transformers=[
            ('', passthrough, remainder_cols),
            ('num', num, num_cols)],
            verbose_feature_names_out=self.VERBOSE_FEATURE_NAMES_OUT)
        self.pipeline = Pipeline(steps=[('transformer', transformer)])

    def fit(self, X, y=None):
        self.pipeline.fit(X)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)


class PassthroughTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class FillBackForward(BaseEstimator, TransformerMixin):
    id_cols = []
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Remove any entries where the school id is not known
        X = X[X['school_id'].notna()]
        # Filling backwards and forwards
        X = self.fill_back_forward(X)
        # Fill remaining NA Values with median
        X = self.fill_median_by_year(X)

        return X

    @staticmethod
    def fill_back_forward(X):
        """ Fills NA values for each school with values from the most recent year first.
        Then fills NA values from the previous year."""
        ids = X[['school_id', 'emh']].unique()
        for i in ids:
            X.loc[X['school_id'] == i] = X.loc[X['school_id'] == i].sort_values(by='year').bfill()
            X.loc[X['school_id'] == i] = X.loc[X['school_id'] == i].sort_values(by='year').ffill()

        return X

    @staticmethod
    def fill_median_by_year(X):
        """ Fills NA values with the median of each year."""
        years = X['year'].unique()
        num_cols = X.select_dtypes(include=('float', 'int')).columns
        for year in years:
            medians = X.loc[X['year'] == year, num_cols].median()
            X.loc[X['year'] == year, num_cols] = X.loc[X['year'] == year, num_cols].fillna(medians)

        return X


class OutlierRemover(BaseEstimator, TransformerMixin):
    """ Removes Outliers using the ECOD class not including the index columns. """
    def __init__(self, index_cols=None, **kwargs):
        # Automatically assume that there are two index columns if none are provided
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


class LLMKMeansPreprocessor(BaseEstimator, TransformerMixin):
    INDEX_COLS = ['school_id', 'year']

    COLS = ['achievement_dir',
            'growth_dir',
            'overall_dir',
            'school_grade',
            'est_total_pop',
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
            'pct_fr']

    HIGH_COLS = ['eng_yn',
                 'math_yn',
                 'read_yn',
                 'sci_yn',
                 'pct_remediation',
                 'graduation_rate']

    def __init__(self, index_cols=None, cols=None, high_school=False, **kwargs):
        # Set defaults if no information was provided
        if index_cols is None:
            self.index_cols = self.INDEX_COLS
        else:
            self.index_cols = index_cols

        if cols is None:
            if not high_school:
                cols = self.COLS
            else:
                cols = self.COLS + self.HIGH_COLS
        else:
            cols = cols

        na_filler = FillBackForward()
        feature_builder = LLMKMeansFeatureBuilder(index_cols=self.index_cols, cols=cols)
        outlier_remover = OutlierRemover(index_cols=self.index_cols)

        steps = [('NA_filler', na_filler),
                 ('feature_builder', feature_builder),
                 ('outlier_remover', outlier_remover)]

        self.pipeline = Pipeline(steps=steps)

    def fit(self, X, y=None):
        self.pipeline.fit(X)
        return self

    def transform(self, X):
        X = self.pipeline.transform(X)
        columns = self.index_cols + [str(i) for i in range(X.shape[1] - 2)]
        return pd.DataFrame(X, columns=columns)


class LLMKMeansFeatureBuilder(BaseEstimator, TransformerMixin):
    """ A Transformer that builds all of the features for the model. """
    VERBOSE_FEATURE_NAMES_OUT = False

    def __init__(self, index_cols=None, cols=None, **kwargs):
        llm = Pipeline(steps=[('encoder', LLM(cols))])
        passthrough = Pipeline(steps=[('encoder', PassthroughTransformer())])
        transformer = ColumnTransformer(transformers=[
            ('', passthrough, index_cols),
            ('num', llm, cols)],
            verbose_feature_names_out=self.VERBOSE_FEATURE_NAMES_OUT)
        self.pipeline = Pipeline(steps=[('transformer', transformer)])

    def fit(self, X, y=None):
        self.pipeline.fit(X)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)


class LLM(BaseEstimator, TransformerMixin):

    def __init__(self, cols, **kwargs):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """ Utilizes Sentence Embedding and BERT to convert all features to the numerical space """
        sentences = X.apply(lambda x: self.compile_text(x, self.cols), axis=1).tolist()

        # The location of the BERT Model
        model = SentenceTransformer(r"sentence-transformers/paraphrase-MiniLM-L6-v2")
        # Process the sentences
        output = model.encode(sentences=sentences,
                              show_progress_bar=True,
                              normalize_embeddings=True)

        # Create an embedding of the output and convert it to a DataFrame
        df_embedding = pd.DataFrame(output)
        return df_embedding

    @staticmethod
    def compile_text(X, cols):
        """ Does a sentence embedding of all the features """
        return ',\n'.join([f'{col}: {X[col]}' for col in cols])