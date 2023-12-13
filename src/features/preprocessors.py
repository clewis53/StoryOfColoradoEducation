import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, OneHotEncoder
from pyod.models.ecod import ECOD
from sentence_transformers import SentenceTransformer

INDEX_COLS = ['unique_id', 'year']

ORD = [
    'achievement_dir',
    'growth_dir',
    'overall_dir',
    'school_grade',
    'overall_weighted_growth'
    # 'read_achievement',
    # 'math_achievement',
    # 'write_achievement',
    # 'science_achievement',
    # 'read_growth',
    # 'math_growth',
    # 'write_growth',

]

ONE_HOT = [
    # 'emh',
    'emh_combined'
]

CAT_COLS = ORD + ONE_HOT

NUM_COLS = [
    'total',
    # 'est_total_pop',
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
    'pct_fr',
]

HIGH_ORD = [
    'eng_yn',
    'math_yn',
    'read_yn',
    'sci_yn'
]

HIGH_ONE_HOT = [
    'emh_combined'
]

HIGH_CAT_COLS = HIGH_ORD + HIGH_ONE_HOT

HIGH_NUM_COLS = [
    'pct_remediation',
    'graduation_rate'
]


class KMeansPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, high_school=False, **kwargs):
        # Set Params
        self.high_school = high_school
        self.index_cols = INDEX_COLS
        self.remainder_cols = INDEX_COLS + ORD
        self.one_hot_cols = ONE_HOT
        self.num_cols = NUM_COLS
        # Adjust params if high_school
        if high_school:
            self.remainder_cols = self.remainder_cols + HIGH_ORD
            self.one_hot_cols = HIGH_ONE_HOT
            self.num_cols = self.num_cols + HIGH_NUM_COLS

        self.all_cols = self.remainder_cols + self.one_hot_cols + self.num_cols

        # Transformers
        imputer = Imputer()
        self.feature_builder = KMeansFeatureBuilder(
            remainder_cols=self.remainder_cols,
            one_hot_cols=self.one_hot_cols,
            num_cols=self.num_cols
        )
        outlier_remover = OutlierRemover(index_cols=self.index_cols)

        # Pipeline Steps
        steps = [
            ('imputer', imputer),
            ('feature_builder', self.feature_builder),
            ('outlier_remover', outlier_remover)
        ]

        # Pipeline
        self.pipeline = Pipeline(steps=steps)

    def fit(self, X, y=None):
        self.pipeline.fit(X)
        return self

    def transform(self, X):
        X = self.pipeline.transform(X)
        onehot_out = []
        if self.one_hot_cols:
            onehot_out = self.feature_builder.transformer.transformers_[1][1].get_feature_names_out().tolist()

        return pd.DataFrame(X, columns=(self.remainder_cols + onehot_out + self.num_cols))


class KMeansFeatureBuilder(BaseEstimator, TransformerMixin):
    VERBOSE_FEATURE_NAMES_OUT = True

    def __init__(self, remainder_cols=None, one_hot_cols=None, num_cols=None, normalization=True, **kwargs):
        # Set Params
        self.remainder_cols = remainder_cols
        self.one_hot_cols = one_hot_cols
        self.num_cols = num_cols
        self.normalization = normalization

        # Transformers
        passthrough = Pipeline(steps=[('encoder', PassthroughTransformer())])
        one_hot = Pipeline(steps=[('encoder', OneHotEncoder(drop='first', sparse_output=False))])
        num = Pipeline(steps=[('encoder', Normalizer())])

        if normalization:
            transformers = [
                ('', passthrough, remainder_cols),
                ('onehot', one_hot, one_hot_cols),
                ('num', num, num_cols)
            ]
        else:
            transformers = [
                ('', passthrough, remainder_cols),
                ('onehot', one_hot, one_hot_cols),
                ('num', passthrough, num_cols)
            ]

        if not one_hot_cols:
            transformers.pop(1)

        self.transformer = ColumnTransformer(
            transformers=transformers,
            verbose_feature_names_out=self.VERBOSE_FEATURE_NAMES_OUT
        )

        self.pipeline = Pipeline(steps=[('transformer', self.transformer)])

    def fit(self, X, y=None):
        self.pipeline.fit(X)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)


class PassthroughTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X):
        self.fit(X)
        return X


class Imputer(BaseEstimator, TransformerMixin):
    id_cols = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Remove any entries where the school id is not known
        X = X[X['unique_id'].notna()]
        # Filling backwards and forwards
        X = self.fill_back_forward(X)
        # Fill remaining NA Values with median
        X = self.fill_by_year(X)

        return X

    @staticmethod
    def fill_back_forward(X):
        """ Fills NA values for each school with values from the most recent year first.
        Then fills NA values from the previous year."""
        ids = X['unique_id'].unique()
        for i in ids:
            X.loc[X['unique_id'] == i] = X.loc[X['unique_id'] == i].sort_values(by='year').bfill()
            X.loc[X['unique_id'] == i] = X.loc[X['unique_id'] == i].sort_values(by='year').ffill()

        return X

    @staticmethod
    def fill_by_year(X):
        """ Fills NA values with the median of each year for numeric and mode for non_numeric."""
        years = X['year'].unique()
        num_cols = X.select_dtypes(include=('float', 'int')).columns
        str_cols = X.select_dtypes(include=('object')).columns
        for year in years:
            medians = X.loc[X['year'] == year, num_cols].median(numeric_only=True)
            modes = X.loc[X['year'] == year, str_cols].mode()

            X.loc[X['year'] == year, num_cols] = X.loc[X['year'] == year, num_cols].fillna(medians)
            X.loc[X['year'] == year, str_cols] = X.loc[X['year'] == year, str_cols].fillna(modes)

        return X


class OutlierRemover(BaseEstimator, TransformerMixin):
    """ Removes Outliers using the ECOD class not including the index columns. """

    def __init__(self, index_cols=None, **kwargs):
        # Automatically assume that there are two index columns if none are provided
        self.index_cols = index_cols

        if index_cols is None:
            self.end_index_cols = 2
        else:
            self.end_index_cols = len(index_cols)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = np.array(X[:, self.end_index_cols:].tolist())
        clf = ECOD()
        clf.fit(data)
        outliers = clf.predict(data)

        X = X[outliers == 0]

        return X


class LLMKMeansPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, high_school=False, **kwargs):
        # Set params
        self.high_school = high_school
        self.index_cols = INDEX_COLS
        self.remainder_cols = INDEX_COLS + ORD
        self.one_hot_cols = ONE_HOT
        self.num_cols = NUM_COLS
        cols = ORD + ONE_HOT + NUM_COLS
        # Adjust params if high_school
        if high_school:
            self.remainder_cols = self.remainder_cols + HIGH_ORD
            self.one_hot_cols = HIGH_ONE_HOT
            self.num_cols = self.num_cols + HIGH_NUM_COLS
            cols = ORD + HIGH_ORD + HIGH_ONE_HOT + NUM_COLS + HIGH_NUM_COLS

        self.all_cols = self.index_cols + cols

        imputer = Imputer()
        feature_builder = LLMKMeansFeatureBuilder(index_cols=self.index_cols, cols=cols)
        outlier_remover = OutlierRemover(index_cols=self.index_cols)

        steps = [('imputer', imputer),
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
        # Set Params
        self.index_cols = index_cols
        self.cols = cols

        # Transformers
        llm = Pipeline(steps=[('encoder', LLM(cols))])
        passthrough = Pipeline(steps=[('encoder', PassthroughTransformer())])

        # Preprocessor
        transformer = ColumnTransformer(transformers=[
            ('', passthrough, index_cols),
            ('num', llm, cols)],
            verbose_feature_names_out=self.VERBOSE_FEATURE_NAMES_OUT)

        # Pipeline
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
        return '\n'.join([f'{col}: {X[col]}' for col in cols])
