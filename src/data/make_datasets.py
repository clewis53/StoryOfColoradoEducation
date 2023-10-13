# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 13:02:25 2023

@author: caeley
"""
from pathlib import Path
from makers import DataFrameSet
import makers
from combine_datasets import combine_datasets
from input_output_functions import append_path, create_filenames


def make_datasets(input_filepath, output_filepath):
    """
    Transforms raw data into usable data saved as interim

    Parameters
    ----------
    output_filepath : str, Path, optional
        The directory to save files in. The default is './'.

    Returns
    -------
    None.

    """
    census = make_census(append_path(input_filepath, 'census'), 
                      append_path(output_filepath, 'census'))
    exp = make_expenditures(append_path(input_filepath, 'expenditures'), 
                            append_path(output_filepath, 'expenditures'))
    kaggle = make_kaggle(append_path(input_filepath,'kaggle'), 
                     append_path(output_filepath,'kaggle'))
    
    # Combine datasets
    change, coact, enroll, final, frl, remediation, address = kaggle    
    combined_datasets = combine_datasets(input_filepath, output_filepath, census, exp, kaggle)
    
    return census, exp, kaggle, combined_datasets
    
    
    



# def make_tall(datasets, id_col=[], id_name='df_id'):
#     """
#     Transforms a list of .csv datasets into a single tall .csv dataset

#     Parameters
#     ----------
#     datasets : list(DataFrame)
#         A non-empty list of DataFrames to combine.
#     id_col: list(int), optional
#         A list of values that should identify each dataframe before concatenating them.
#         The default is [] or don't add id_col.
#     id_name: String, optional
#         The name for the id column to be used when id_cols is not empty.
#         The default is 'df_id'.

#     Returns
#     -------
#     tall_df: DataFrame
#         The dataframe of the tall version of the given datasets

#     """
#     # Initialize DataFrame
#     tall_df = datasets[0]   
#     # Make changes if id_col have been added
#     if id_col:
#         # The length of id_col must be equal to the number of datasets provided
#         assert len(id_col) == len(datasets)
#         # Their must be an id_name given when an id_col is specified
#         assert id_name != None
#         # Add the id column
#         tall_df[id_name] = id_col[0]
    
#     # Iterate over all datasets and id_col
#     for i in range(1, len(datasets)):
#         # Add the id column if necessary
#         if id_col:
#             datasets[i][id_name] = id_col[i]

#         tall_df = pd.concat((tall_df, datasets[i]))
        
#     return tall_df



def make_census(input_filepath, output_filepath, years=(2010, 2011, 2012)):
    """
    Transforms raw census data into usable tall interim data.
    The input filepath must contain saipe datasets that
    that were downloaded using get_raw_data module.
    Finally, it saves the tall_df in the output_filepath
    
    Parameters
    ----------
    input_filepath : str, Path
        the directory to obtain files from
    output_filepath : str, Path
        the directory to save files in

    Returns
    -------
    None.
    """
    # Input and output locations
    input_filenames = create_filenames(input_filepath, 'saipe{year}.csv')
    output_filenames = create_filenames(output_filepath, 'saipe{year}.csv')
    
    # MakeDatasets
    dataframes = DataFrameSet(input_filenames, output_filenames, makers.CensusMaker)
    dataframes.make_dataframes()
    
    tall_filepath = append_path(output_filepath, 'tall_saipe.csv')
    return dataframes.make_tall(id_col=years, filepath=tall_filepath)
    



def make_expenditures(input_filepath, output_filepath, years=(2010, 2011, 2012)):
    """
    Transforms all expenditures datasets that must be Comparison of All 
    Program Expenditures (All Funds) directly downloaded from
    https://www.cde.state.co.us/cdefinance/RevExp and then saved as 
    expenditures{year}.csv
    
    Parameters
    ----------
    input_filepath : str, Path
        the directory to obtain files from
    output_filepath : str, Path
        the directory to save files in

    Returns
    -------
    None.

    """
    # Input and output locations    
    input_filenames = create_filenames(input_filepath, 'expenditures{year}.csv')
    output_filenames = create_filenames(output_filepath, 'expenditures{year}.csv')
    
    # Make datasets
    datasets = DataFrameSet(input_filenames, output_filenames, makers.ExpenditureMaker)
    datasets.make_dataframes()
    
    tall_filepath = append_path(output_filepath, 'tall_expenditures.csv')
    return datasets.make_tall(id_col=years, filepath=tall_filepath)
    
    



def make_kaggle(input_filepath, output_filepath, years=(2010, 2011, 2012)):
    """
    Transforms each kaggle raw dataset into individual usable tall interim data
    
    Parameters
    ----------
    input_filepath : str, Path
        the directory to obtain files from
    output_filepath : str, Path
        the directory to save files in

    Returns
    -------
    None.

    """
    
    change = make_1yr_3yr_change(input_filepath, output_filepath)
    coact = make_coact(input_filepath, output_filepath)
    enroll = make_enrl_working(input_filepath, output_filepath)
    final = make_final_grade(input_filepath, output_filepath)
    frl = make_k_12_frl(input_filepath, output_filepath)
    remediation = make_remediation(input_filepath, output_filepath)
    address = make_school_address(input_filepath, output_filepath)
    
    return change, coact, enroll, final, frl, remediation, address

def make_1yr_3yr_change(input_filepath, output_filepath, years=(2010, 2011, 2012)):
    """
    Transforms 1yr_3yr_change datasets downloaded from the kaggle competition

    Parameters
    ----------
    input_filepath : String, Path
        The input filepath base to extract data from
    output_filepath : String, Path
        The output filepath base to save data to

    Returns
    -------
    None.

    """
    
    input_filenames = create_filenames(input_filepath, '{year}_1YR_3YR_change.csv')
    output_filenames = create_filenames(output_filepath, '1YR_3YR_change{year}.csv')
    
    datasets = DataFrameSet(input_filenames, output_filenames, makers.ChangeMaker)
    datasets.make_dataframes()
    
    tall_filepath = append_path(output_filepath, '1YR_3YR_change_tall.csv')
    return datasets.make_tall(id_col=years, filepath=tall_filepath)
    

def make_coact(input_filepath, output_filepath, years=(2010, 2011, 2012)):
    input_filenames = create_filenames(input_filepath, '{year}_COACT.csv')    
    output_filenames = create_filenames(output_filepath, 'COACT{year}.csv')
    
    datasets = DataFrameSet(input_filenames, output_filenames, makers.CoactMaker)
    datasets.make_dataframes()
    
    tall_filepath = append_path(output_filepath, 'COACT_tall.csv')
    return datasets.make_tall(id_col=years, filepath=tall_filepath)
    
def make_enrl_working(input_filepath, output_filepath, years=(2010, 2011, 2012)):
    input_filenames = create_filenames(input_filepath, '{year}_enrl_working.csv')    
    output_filenames = create_filenames(output_filepath, 'enrl_working{year}.csv')
    
    datasets = DataFrameSet(input_filenames, output_filenames, makers.EnrollMaker)
    datasets.make_dataframes()
    
    tall_filepath = append_path(output_filepath, 'enrl_working_tall.csv')
    return datasets.make_tall(id_col=years, filepath=tall_filepath)
    

def make_final_grade(input_filepath, output_filepath, years=(2010, 2011, 2012)):
    input_filenames = create_filenames(input_filepath, '{year}_final_grade.csv')      
    output_filenames = create_filenames(output_filepath, 'final_grade{year}.csv')    

    datasets = DataFrameSet(input_filenames, output_filenames, makers.FinalMaker)
    datasets.make_dataframes()
    
    tall_filepath = append_path(output_filepath, 'final_grade_tall.csv')
    return datasets.make_tall(id_col=years, filepath=tall_filepath)


def make_k_12_frl(input_filepath, output_filepath, years=(2010, 2011, 2012)):
    input_filenames = create_filenames(input_filepath, '{year}_k_12_FRL.csv')        
    output_filenames = create_filenames(output_filepath, 'FRL{year}.csv')
    
    datasets = DataFrameSet(input_filenames, output_filenames, makers.FrlMaker)
    datasets.make_dataframes()
    
    tall_filepath = append_path(output_filepath, 'FRL_tall.csv')
    return datasets.make_tall(id_col=years, filepath=tall_filepath)


def make_remediation(input_filepath, output_filepath, years=(2010, 2011, 2012)):
    
    input_filenames = create_filenames(input_filepath, '{year}_remediation_HS.csv')      
    output_filenames = create_filenames(output_filepath, 'remediation{year}.csv')
        
    datasets = DataFrameSet(input_filenames, output_filenames, makers.RemediationMaker)
    datasets.make_dataframes()
    
    tall_filepath = append_path(output_filepath, 'remediation_tall.csv')
    return datasets.make_tall(id_col=years, filepath=tall_filepath)


def make_school_address(input_filepath, output_filepath, years=(2010, 2011, 2012)):
    input_filenames = create_filenames(input_filepath, '{year}_school_address.csv')    
    output_filenames = create_filenames(output_filepath, 'address{year}.csv')
    
    datasets = DataFrameSet(input_filenames, output_filenames, makers.AddressMaker)
    datasets.make_dataframes()
    
    tall_filepath = append_path(output_filepath, 'address_tall.csv')
    return datasets.make_tall(id_col=years, filepath=tall_filepath)
    


def main(input_filepath, output_filepath):
    make_datasets(input_filepath, output_filepath)


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]    
    input_filepath = project_dir.joinpath("data/raw")
    output_filepath = project_dir.joinpath("data/interim")
    
    main(input_filepath, output_filepath)
