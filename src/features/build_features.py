import pandas as pd
from pathlib import Path
from src.features.preprocessors import KMeansPreprocessor, LLMKMeansPreprocessor
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
    processed_all = preprocessor.fit_transform(all_data)
    processed_all.to_csv(append_path(output_filepath, 'kmeans_all_data.csv'), index=False)

    # Processes the high_school data and saves it
    preprocessor = KMeansPreprocessor(high_school=True)
    processed_high = preprocessor.fit_transform(high_school)
    processed_high.to_csv(append_path(output_filepath, 'kmeans_high_school.csv'), index=False)

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
    processed_all = preprocessor.fit_transform(all_data)
    processed_all.to_csv(append_path(output_filepath, 'llc_kmeans_all_data.csv'), index=False)

    # Processes the high_school data and saves it
    preprocessor = LLMKMeansPreprocessor(high_school=True)
    processed_high = preprocessor.fit_transform(high_school)
    processed_high.to_csv(append_path(output_filepath, 'llc_kmeans_high_school.csv'), index=False)

    # Returns the processed dataframes
    return processed_all, processed_high


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[2]
    input_filepath = project_dir.joinpath("data/interim")
    output_filepath = project_dir.joinpath("data/processed")

    main(input_filepath, output_filepath)
