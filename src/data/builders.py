# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 19:36:49 2023

@author: caeley
"""
import pandas as pd

# List of changes to make to district_name
DISTRICT_NAME_CHANGES = {' SCHOOLS': '',
                         'SCHOOL ': '',
                         'DISTRICT ': '',
                         'DISTRICT': '',
                         '-': ' ',
                         ':': ' ',
                         'S/D ': '',
                         '/': ' ',
                         r'[^\w\s]+': '',
                         'RURAL ': '',
                         'NO2': '2',
                         '29J': '29 J',
                         '49JT':'49 JT',
                         'RE1J':'RE 1J',
                         'C113': 'C 113',
                         'MILIKEN': 'MILLIKEN',
                         'MC CLAVE': 'MCCLAVE',
                         'NO 1': '1',
                         'PARK ESTES PARK': 'PARK',
                         'PARK R 3': 'ESTES PARK R 3',
                         'WELD RE 1': 'WELD COUNTY RE 1',
                         'MOFFAT 2': 'MOFFAT COUNTY 2',
                         '10 JT R': 'R 10 JT',
                         'GARFIELD 16': 'GARFIELD COUNTY 16',
                         'MOFFAT CONSOLIDATED': 'MOFFAT COUNTY',
                         'Ñ': 'N',
                         'NORTHGLENN THORNTON 12': 'ADAMS 12 FIVE STAR',
                         'NORTHGLENN-THORNTON 12': 'ADAMS 12 FIVE STAR',
                         'CONSOLIDATED C 1': 'CUSTER COUNTY C 1',
                         'CUSTER COUNTY DISTR': 'CUSTER COUNTY C 1',
                         'CREEDE CONSOLIDATED 1': 'CREEDE',
                         'FLORENCE': 'FREMONT'
                        }

class IDDatasetBuilder:
    """ Base Class that helps build ID datasets from a collection of datasets that
        contain pieces of information about the entire list of ids"""
    # The columns that are pertinent to the ID Dataset
    keep_cols = []
    # Save the id_cols to search duplicates for
    id_cols = []
    
    def __init__(self, kaggle_datasets):
        """
        
        Parameters
        ----------
        kaggle_datasets : list(pd.DataFrames)
            A list of kaggle dataframes to build an id dataset from

        Returns
        -------
        None.

        """
        # Save only pertinent information of datasets
        self.datasets = [dataset[self.keep_cols] for dataset in kaggle_datasets]
        # Initialize id_dataset
        self.id_dataset = pd.DataFrame()
    
        
    def build(self):
        """
        Concatenates a list of id data then drops duplicates keeping the first entry
        """
        # Create a long list of id_datasets
        self._concatenate()
        
        # Remove duplicates
        self.id_dataset = self.id_dataset.drop_duplicates(subset=self.id_cols)

        # Remove NA
        self.id_dataset = self.id_dataset.dropna(subset=self.id_cols)
        
                 
    def save(self, filepath):
        """ Saves the ID dataset to the given filepath """
        self.id_dataset.to_csv(filepath, index=False)
    
    
    def _concatenate(self):
        """ Concatenate a list of ids """
        # Initialize the id_dataset
        self.id_dataset = self.datasets[0]
        
        for i in range(1, len(self.datasets)):
            self.id_dataset = pd.concat((self.id_dataset, self.datasets[i]))
            
    
class SchoolIDBuilder(IDDatasetBuilder):
    # The columns that make the schools dataset
    keep_cols = ['school_id', 'emh', 'school', 'district_id']
    # The id column for the school id dataset
    id_cols = ['school_id', 'emh']

    def build(self):
        super().build()
        self._create_unique_id()

    def _create_unique_id(self):
        self.id_dataset['unique_id'] = self.id_dataset['school_id'].astype('int').astype('string') + self.id_dataset['emh']
    

class DistrictIDBuilder(IDDatasetBuilder):
    # The columns that make the districts dataset
    keep_cols = ['district_id', 'district_name']
    # The id column in the districts dataset
    id_cols = ['district_id']
    
    def build(self):
        super().build()
        self._transform_district_name()
    
        
    def _transform_district_name(self):
        self.id_dataset['district_name'] = transform_district_name(self.id_dataset['district_name'])
    
    
def transform_district_name(col):
    # Uppercase the district_names
    col = col.str.upper()
    
    # Apply all changes
    for original, replacement in DISTRICT_NAME_CHANGES.items():
        col = col.str.replace(original, replacement, regex=True)
    
    col = col.str.strip()
        
    return col
