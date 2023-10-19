# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 18:35:03 2023

@author: caeley
"""
from src.input_output_functions import append_path
import pandas as pd
from pathlib import Path
import src.data.builders as builders


def combine_datasets(input_filepath, output_filepath, census, exp, kaggle):
    # Extract kaggle datasets
    change, coact, enroll, final, frl, remediation, address = kaggle
    
    # Build district dataset
    district = create_district_dataset(input_filepath, output_filepath,
                                       change, enroll, final, frl)
    # Build school dataset
    school = create_school_dataset(input_filepath, output_filepath,
                                   change, final)
    
    census, exp = find_district_id(district, census, exp)
    
    remove_bad_info_datasets = census, exp, change, coact, enroll, final, frl, remediation,
    # Update datasets by removing district and school information
    updated_datasets = remove_district_and_school_info(remove_bad_info_datasets, district, school)
    # Extract updated datasets
    census, exp, change, coact, enroll, final, frl, remediation = updated_datasets
      
    # Build all data
    all_data = create_all_data(input_filepath, output_filepath,
                               census, 
                               exp, 
                               change, enroll, final, frl,
                               district, school)
    
    # Build high school data
    high_school = create_high_school(input_filepath, output_filepath, coact, remediation, all_data)
    
    # Build the GPS location dataset
    gps = pd.DataFrame()
    create_gps_location(input_filepath, output_filepath, gps, school)
    
    return district, school, all_data, high_school
    
    
def create_district_dataset(input_filepath, output_filepath, 
                            change, enroll, final, frl):
    
    district_builder = builders.DistrictIDBuilder((change, enroll, final, frl))
    district_builder.build()
    district_builder.save(append_path(output_filepath, 'districts.csv'))
    
    return district_builder.id_dataset
    

def create_school_dataset(input_filepath, output_filepath, 
                          change, final):
    kaggle_datasets = change, final
    
    school_builder = builders.SchoolIDBuilder(kaggle_datasets)
    school_builder.build()
    school_builder.save(append_path(output_filepath, 'schools.csv'))

    
    return school_builder.id_dataset


def create_all_data(input_filepath, output_filepath, 
                    census, 
                    exp, 
                    change, enroll, final, frl,
                    district, school):
    census_exp_df = pd.merge(census, exp, on=['district_id', 'year'], how='outer')
    change_final_df = pd.merge(change, final, on=['school_id', 'district_id', 'emh', 'year'], how='outer')
    
    all_data = pd.merge(census_exp_df, change_final_df, on=['district_id', 'year'], how='outer')
    all_data = pd.merge(all_data, enroll.drop('district_id', axis=1), on=['school_id', 'year'], how='outer')
    all_data = pd.merge(all_data, frl.drop('district_id', axis=1), on=['school_id', 'year'], how='outer')
    all_data = pd.merge(all_data, district, on='district_id')
    all_data = pd.merge(all_data, school, on=['school_id', 'emh', 'district_id'], how='outer')

    all_data.drop('graduation_rate', axis=1).to_csv(append_path(output_filepath, 'all_data.csv'), index=False)
    
    return all_data
    
def create_high_school(input_filepath, output_filepath,
                       coact, remediation,
                       all_data):
    coact_remediation = pd.merge(coact, remediation, on=['school_id', 'year'])
    all_data_high_schools = all_data[all_data['emh'] == 'H'].drop('emh', axis=1)

    high_school = pd.merge(coact_remediation, all_data_high_schools, on=['school_id', 'district_id', 'year'])
    
    high_school.to_csv(append_path(output_filepath, 'high_school.csv'), index=False)
    
    return high_school


def create_gps_location(input_filepath, output_filepath, gps, school):
    pass

def find_district_id(district, census, exp):
    def _find_district_id(df):
        df['district_name'] = builders.transform_district_name(df['district_name'])
        return pd.merge(district, df, on='district_name')
        
    return _find_district_id(census), _find_district_id(exp)


def remove_district_and_school_info(datasets, districts, schools):
    remove_cols = ['district_name', 'school']
    for i in range(len(datasets)):
        datasets[i].drop(remove_cols, axis=1, errors='ignore', inplace=True)
        
    return datasets


def add_district_id(dataset):
    if 'district_name' not in dataset.columns:
        raise ValueError('district_name must be a column in the dataset')


def main(input_filepath, output_filepath):
    # census
    census = pd.read_csv(append_path(input_filepath, 'census/tall_saipe.csv'))
    # expenditures
    exp = pd.read_csv(append_path(input_filepath, 'expenditures/tall_expenditures.csv'))
    # kaggle
    change = pd.read_csv(append_path(input_filepath, 'kaggle/1YR_3YR_change_tall.csv'))
    coact = pd.read_csv(append_path(input_filepath, 'kaggle/COACT_tall.csv'))
    enroll = pd.read_csv(append_path(input_filepath, 'kaggle/enrl_working_tall.csv'))
    final = pd.read_csv(append_path(input_filepath, 'kaggle/final_grade_tall.csv'))
    frl = pd.read_csv(append_path(input_filepath, 'kaggle/FRL_tall.csv'))
    remediation = pd.read_csv(append_path(input_filepath, 'kaggle/remediation_tall.csv'))
    address = pd.DataFrame()
    
    kaggle = change, coact, enroll, final, frl, remediation, address

    
    combine_datasets(input_filepath, output_filepath, census, exp, kaggle)


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]    
    input_filepath = project_dir.joinpath("data/interim")
    output_filepath = project_dir.joinpath("data/interim")
    
    main(input_filepath, output_filepath)
