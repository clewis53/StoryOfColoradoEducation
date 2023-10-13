import pandas as pd
from pathlib import Path
from preprocessors import KMeansPreprocessor


def main(input_filepath, output_filepath):
    all_interim_df = pd.read_csv(input_filepath.joinpath('all_data.csv'))
    high_interim_df = pd.read_csv(input_filepath.joinpath('high_school.csv'))

    build_kmeans(all_interim_df, high_interim_df, output_filepath)


def build_kmeans(all_data, high_school, output_filepath):
    """ Builds a dataframe that will be used for a kmeans model. The preprocessor used is in preprocessors.KMeansPreprocessor"""
    preprocessor = KMeansPreprocessor()
    processed_df = preprocessor.fit_transform(all_data)
    processed_df.to_csv(output_filepath.joinpath('all_data.csv'), index=False)

    preprocessor = KMeansPreprocessor(high_school=True)
    processed_df = preprocessor.fit_transform(high_school)
    processed_df.to_csv(output_filepath.joinpath('high_school.csv'), index=False)


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[2]
    input_filepath = project_dir.joinpath("data/interim")
    output_filepath = project_dir.joinpath("data/processed")

    main(input_filepath, output_filepath)
