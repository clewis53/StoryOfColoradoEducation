import pandas as pd
from pathlib import Path
from src.features.preprocessors import KMeansPreprocessor, LLMKMeansPreprocessor, FillBackForward
from src.input_output_functions import append_path


def main(input_filepath, output_filepath):
    all_interim_df, high_interim_df = load_interim_data(input_filepath)

    build_kmeans(output_filepath, all_interim_df, high_interim_df)

    build_llm_kmeans(output_filepath, all_interim_df, high_interim_df)


def load_interim_data(input_filepath):
    """ Loads the interim data located at the input_filepath.
    File format must be all_data.csv and high_school.csv"""
    all_data = pd.read_csv(append_path(input_filepath, 'all_data.csv'))
    high_school = pd.read_csv(append_path(input_filepath, 'high_school.csv'))

    return all_data, high_school


def get_no_outliers(original_df, processed_df, preprocessor):
    """ Merges the original and processed data frames"""
    # Identify the columns that were selected after processing
    correct_cols = original_df[preprocessor.all_cols]

    no_outliers = pd.merge(processed_df[preprocessor.index_cols], correct_cols, on=preprocessor.index_cols, how='left')
    na_filler = FillBackForward()

    return na_filler.fit_transform(no_outliers)

def process_data(df, preprocessor, output_filepath, filename):
    """ Processes data and saves it along with the original data that has been updated to not include outliers """
    processed_df = preprocessor.fit_transform(df)
    no_outliers = get_no_outliers(df, processed_df, preprocessor)

    processed_df.to_csv(append_path(output_filepath, filename), index=False)
    no_outliers.to_csv(append_path(output_filepath, 'no_outliers_' + filename), index=False)

    print('finished processing')

    return processed_df


def build_kmeans(output_filepath, all_data=None, high_school=None, input_filepath=None):
    """ Builds a dataframe from all_data and high_school that will be used for a kmeans model, saves them, and returns them.
    The preprocessor used is in preprocessors.KMeansPreprocessor """

    # Checks to see if all_data or high_school were not provided
    if all_data is None or high_school is None:
        # if they weren't makes sure that an input_filepath was provided
        if input_filepath is None:
            raise ValueError('If input_filepath is not provided, all_data and high_school must both be provided.')
        # loads and replaces all_data, and high_school if necessary
        all_data, high_school = load_interim_data(input_filepath)

    # Processed the all_data and saves it
    preprocessor = KMeansPreprocessor()
    processed_all = process_data(all_data, preprocessor, output_filepath, 'kmeans_all_data.csv')

    # Processes the high_school data and saves it
    preprocessor = KMeansPreprocessor(high_school=True)
    processed_high = process_data(high_school, preprocessor, output_filepath, 'kmeans_high_school.csv')

    # Returns the processed dataframes
    return processed_all, processed_high


def build_llm_kmeans(output_filepath, all_data=None, high_school=None, input_filepath=None):
    """ Builds a dataframe from all_data and high_school that will be used for a llm processed kmeans model,
    saves them, and returns them.
    The preprocessor used is in preprocessors.LLCKMeansPreprocessor """

    # Checks to see if all_data or high_school were not provided
    if all_data is None or high_school is None:
        # if they weren't makes sure that an input_filepath was provided
        if input_filepath is None:
            raise ValueError('If input_filepath is not provided, all_data and high_school must both be provided.')
        # loads and replaces all_data, and high_school if necessary
        all_data, high_school = load_interim_data(input_filepath)

    # Processed the all_data and saves it
    preprocessor = LLMKMeansPreprocessor()
    processed_all = process_data(all_data, preprocessor, output_filepath, 'llm_kmeans_all_data.csv')

    # Processes the high_school data and saves it
    preprocessor = LLMKMeansPreprocessor(high_school=True)
    processed_high = process_data(high_school, preprocessor, output_filepath, 'llm_kmeans_high_school.csv')

    # Returns the processed dataframes
    return processed_all, processed_high


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[2]
    input_filepath = project_dir.joinpath("data/interim")
    output_filepath = project_dir.joinpath("data/processed")

    main(input_filepath, output_filepath)
