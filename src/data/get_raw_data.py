# -*- coding: utf-8 -*-
from pathlib import Path

# Access environment variables
from dotenv import load_dotenv
import os
load_dotenv()

# Census API
import requests # to request census data
from requests import HTTPError
import pandas as pd # to save api json as csv

# Kaggle API
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile # to unzip all files from kaggle


# Census params
CENSUS_YEARS = 2010, 2011, 2012 # The years to request census data
# The parameters needed to request school district name, estimated number of children in poverty,
# estimated number of children, and estimated total population for Colorado 
CENSUS_PARAMS = {'get': 'SD_NAME,SAEPOV5_17RV_PT,SAEPOV5_17V_PT,SAEPOVALL_PT',
                 'time': 999,
                 'for': 'school district (unified)',
                 'in': 'state:08'}

# Kaggle params
COMPETITION_NAME = 'visualize-the-state-of-education-in-colorado'


def append_path(path, addition):
    """
    Appends the path to a string based on type.

    Parameters
    ----------
    path : str, Path
        The original path
    addition : String
        The string to append to the path

    Returns
    -------
    str, Path
        The new path

    """
    if type(path) == str:
        return path + '/' + addition
    return path.joinpath(addition)


def get_census(output_filepath):
    """
    Obtains census data from the its api. 
    For more information, refer to https://api.census.gov/data/timeseries/poverty/saipe/schdist.html

    Parameters
    ----------
    output_filepath : str, Path
        the directory to save files in.

    Returns
    -------
    None.

    """
    # Census API params
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables

    url = os.getenv('CENSUS_URL') # url to request saipe info
    key = os.getenv('CENSUS_KEY')
    
    CENSUS_PARAMS['key'] = key
    # Create a dataframe for each year requested
    for time in CENSUS_YEARS:
        CENSUS_PARAMS['time'] = time # establish time parameter
        with requests.get(url, CENSUS_PARAMS) as response:
            try:
                # check for correct response code
                response.raise_for_status()
                
                # obtain json and turn it into a DataFrame using the first row as column names
                data_json = response.json()
                df = pd.DataFrame(data=data_json[1:], columns=data_json[0])
                
                # Save file
                filename = append_path(output_filepath, f'saipe{time}.csv')
                df.to_csv(filename)
            
            # catches failed response codes
            except HTTPError:
                print(f'{time=} request failed with {response.status_code=}')
            
        
def get_kaggle(output_filepath):
    """
    Obtains all data from the kaggle competition using its api and unzips it.
    For more information, refer to https://www.kaggle.com/docs/api
    
    Parameters
    ----------
    output_filepath : str, Path
        The directory to save files in.

    Returns
    -------
    None.

    """
    KAGGLE_key = os.getenv('KAGGLE_key')
    KAGGLE_username = os.getenv('KAGGLE_username')
    # Create a connection to the kaggle api
    api = KaggleApi()
    api.authenticate()
    
    # Download all competition files in a zip file
    api.competition_download_files(COMPETITION_NAME, output_filepath)   
      
    # Extract all files
    filename = append_path(output_filepath, COMPETITION_NAME + '.zip')
    with zipfile.ZipFile(filename, 'r') as zipref:
        zipref.extractall(output_filepath)
    

def main(output_filepath):
    """
    Makes requests to download files and save them in project_dir/data/raw

    Parameters
    ----------
    output_filepath : str, Path
        The directory to save files in.

    Returns
    -------
    None.

    """
    get_census(output_filepath.joinpath('census'))
    get_kaggle(output_filepath.joinpath('kaggle'))    



if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]    
    output_filepath = project_dir.joinpath("data/raw")
    
    main(output_filepath)
