# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 10:29:27 2023

@author: caeley
"""
import numpy as np
import pandas as pd


class DataFrameSet:
    """ Class to get transform and save sets of dataframes """

    def __init__(self, input_filenames, output_filenames, maker):
        if len(input_filenames) != len(output_filenames):
            raise ValueError(f'input_filenames {len(input_filenames)=}',
                             f'is not the same {len(output_filenames)=}')
        if Maker not in maker.mro():
            raise TypeError('maker must be of type Maker')

        self.input_filenames = input_filenames
        self.output_filenames = output_filenames
        self.maker = maker
        # Initialize dataframes as an empty array of dataframes
        self.dataframes = [pd.DataFrame([])] * len(input_filenames)

    def make_dataframes(self):
        self._get_dataframes()
        self._transform_dataframes()
        self._save_dataframes()

    def _get_dataframes(self):
        """ Reads in all dataframes from the input_filenames iterable"""
        for i in range(len(self.input_filenames)):
            self.dataframes[i] = pd.read_csv(self.input_filenames[i])

    def _transform_dataframes(self):
        """ Transforms all filenames according to the maker class """
        for i in range(len(self.dataframes)):
            df_maker = self.maker(self.dataframes[i])
            df_maker.transform()
            # Reset the value to the transformed version
            self.dataframes[i] = df_maker.df

    def _save_dataframes(self):
        """ Saves all dataframes to the output location """
        for i in range(len(self.output_filenames)):
            self.dataframes[i].to_csv(self.output_filenames[i], index=False)

    def make_tall(self, id_col=(2010, 2011, 2012), id_name='year', filepath=None):

        # The length of id_col must be equal to the number of datasets provided
        if len(id_col) != len(self.dataframes):
            raise ValueError(f'Length of id_col must be {len(self.dataframes)}')
        # Their must be an id_name given when an id_col is specified
        if id_name == None:
            raise ValueError('id_name must not be None')

        # Initialzie the tall_dataframe as the first in the set
        tall_df = self.dataframes[0]
        # Create the id_col to distinguish the dataframes when they are concatenated
        tall_df[id_name] = id_col[0]

        # Concatenate the dataframes
        for i in range(1, len(self.dataframes)):
            self.dataframes[i][id_name] = id_col[i]
            tall_df = pd.concat((tall_df, self.dataframes[i]))

        # Save the dataframe when filepath is not None
        if filepath is not None:
            tall_df.to_csv(filepath, index=False)

        return tall_df


class Maker:
    """ A class that transforms a dataframe into a readable format """

    # Rows that should be dropped from the dataset
    drop_rows = []
    # Columns that should be dropped
    drop_cols = []
    # How to rename columns
    col_map = {}

    def __init__(self, dataframe):
        self.df = dataframe

    def transform(self):
        """ Main function that performs the transformation """
        self._transform_rows()
        self._transform_cols()

    def _transform_rows(self):
        """ Helper function to transform rows """
        # Rows
        # Drop specified rows
        self.df = self.df.drop(self.drop_rows)
        # and any completely empty rows
        self.df = self.df.dropna(how='all')
        # Then reset the index
        self.df = self.df.reset_index(drop=True)

    def _transform_cols(self):
        """ Helper function to transform columns """
        # Columns
        # Lowercase column names
        self.df.columns = self.df.columns.str.lower()
        # Rename columns
        self.df = self.df.rename(columns=self.col_map)
        # Drop columns
        self.df = self.df.drop(self.drop_cols, axis=1, errors='ignore')
        # Infer Column types
        self.df = self.df.convert_dtypes()


class CensusMaker(Maker):
    col_map = {'sd_name': 'district_name',
               'saepov5_17rv_pt': 'est_child_poverty',
               'saepov5_17v_pt': 'est_total_child',
               'saepovall_pt': 'est_total_pop',
               'time': 'year'}
    drop_cols = ['state', 'school district (unified)']

    def transform(self):
        # One change is made to ensure that the index is set properly before transofmring
        self.df = self.df.set_index(self.df.columns[0])
        super().transform()
        self._create_ratio_cols()

    def _create_ratio_cols(self):
        self.df['child_pov_ratio'] = self.df['est_child_poverty'] / self.df['est_total_child']
        self.df['child_adult_ratio'] = self.df['est_total_child'] / (
                    self.df['est_total_pop'] - self.df['est_total_child'])


class ExpenditureMaker(Maker):
    col_map = {'unnamed: 1': 'county',
               'district/': 'district_name',
               'unnamed: 3': 'instruction',
               'total': 'support',
               'unnamed: 5': 'community',
               'unnamed: 6': 'other',
               'unnamed: 7': 'sum'}
    drop_cols = ['unnamed: 0']

    def transform(self):
        # Apply standard changes
        super().transform()
        # Apply additional changes
        self._clean_numbers()
        self._clean_district_name()
        self._extract_data()

    def _clean_numbers(self):
        """ Helper function that deals with all columns of type string. It
            commas, and parantheses
        """

        for col in self.df.columns:
            if self.df[col].dtype == 'string':
                self.df[col] = self.df[col].str.replace(',', '', regex=True)
                self.df[col] = self.df[col].str.replace('\(', '', regex=True)
                self.df[col] = self.df[col].str.replace('\)', '', regex=True)

    def _clean_district_name(self):
        """ Helper function that removes any district names that were BOCES """

        # The district_name column has numbers that were relevant to the BOCES funding but not our project.
        # We want to be able to identify each of those and remove them.
        def remove_floats(entry):
            try:
                float(entry)
                return np.nan
            except:
                return entry

        self.df['district_name'] = self.df['district_name'].apply(remove_floats)

        # Now that they are removed, lets forward fill the district_name,
        # so that we can extract the total amount for each category
        # and the per pupil amount for each category
        self.df['district_name'] = self.df['district_name'].fillna(method='ffill').fillna(method='bfill')

        # Remove all BOCES funding entries because they are irrelevant
        self.df = self.df[~(self.df['district_name'].str.lower().str.contains('boces'))]

    def _extract_data(self):
        """ Helper function that extracts the data from their respective rows and
            organizes it in a column format """

        # Now we can extract the total amounts
        totals = self.df[self.df['county'].str.lower() == 'amount'].drop('county', axis=1)
        # the per pupil amounts 
        per_pupils = self.df[self.df['county'].str.lower() == 'per pupil'].drop('county', axis=1)
        # and the county names
        counties = self.df.loc[
            ~(self.df['county'].str.lower().isin(('amount', 'per pupil', 'all funds', 'county'))), ['district_name',
                                                                                                    'county']].dropna()

        # Now we can merge them
        merged_df = pd.merge(left=totals, right=per_pupils, on='district_name', suffixes=('_total', '_per_pupil'))
        self.df = pd.merge(left=counties, right=merged_df, on='district_name')
        self.df = self.df.convert_dtypes()


class KaggleMaker(Maker):
    col_map = {'emh-combined': 'emh_combined',
               'emh_combined': 'emh_combined',
               'spf_emh_code': 'emh',
               'spf_included_emh_for_a': 'emh_combined',
               'district code': 'district_id',
               'districtnumber': 'district_id',
               'district no': 'district_id',
               'district number': 'district_id',
               'districtnumber': 'district_id',
               'spf_dist_number': 'district_id',
               'org. code': 'district_id',
               'organization code': 'district_id',
               'districtname': 'district_name',
               'district name': 'district_name',
               'spf_district_name': 'district_name',
               'organization name': 'district_name',
               'school_district': 'district_name',
               'school_districte': 'district_name',
               'school_name': 'school',
               'schoolname': 'school',
               'school name': 'school',
               '2010 school name': 'school',
               'spf_school_name': 'school',
               'schoolnumber': 'school_id',
               'school code': 'school_id',
               'school number': 'school_id',
               'school no': 'school_id',
               'spf_school_number': 'school_id'}

    def transform(self):
        # Perform standard changes
        super().transform()

        # Remove rows with unecessary information
        self._remove_boces()
        self._remove_state_results()
        self._remove_district_results()
        # Reset the index after changing rows
        self.df = self.df.reset_index(drop=True)

        # Remove columns with unecessary information
        # self._remove_emh_combined()

        # Refactor columns
        self._refactor_emh_combined()

    def _refactor_emh_combined(self):
        """ 
        Refactors the EMH Combined column to a true false indicator
        of whether or not the school is an EMH combined school
        """
        if 'emh_combined' in self.df.columns:
            self.df['emh_combined'] = self.df['emh_combined'].notna()

    def _remove_boces(self):
        """
        Removes any rows from a dataset where the district_name contains BOCES.
        """
        if 'district_name' in self.df.columns:
            boces_loc = (self.df['district_name'].str.upper().str.contains('BOC'))
            self.df = self.df.drop(self.df[boces_loc].index)

    def _remove_district_results(self):
        """ Removes any rows that contain information for the entire district. """

        # The locations of the district result rows
        dist_loc = self.df['school'] == 'DISTRICT RESULTS'

        self.df = self.df.drop(self.df[dist_loc].index)

    def _remove_emh_combined(self):
        """ Removes the emh_combined column if it exists. """

        if 'emh_combined' in self.df.columns:
            self.df = self.df.drop('emh_combined', axis=1)

    def _remove_state_results(self):
        """ Removes any rows containing state results where the district_id is 0. """

        if 'district_id' in self.df.columns:
            # The location of state results rows
            state_loc = self.df['district_id'] == 0

            self.df = self.df.drop(self.df[state_loc].index)

    def _clean_pct_signs(self, col):
        """ Removes percent signs from the specified column and ignores errors """
        try:
            self.df[col] = self.df[col].str.replace('%', '').astype('float') / 100
        except AttributeError:
            pass


class ChangeMaker(KaggleMaker):
    change_col_map = {'rate_at.5_chng_ach': 'achievement_dir',
                      'rate_at.5_chng_gro': 'growth_dir',
                      'rate_at.5_chng_growth': 'growth_dir',
                      'pct_pts_chng_.5': 'overall_dir',
                      'pct_pts_chnge_.5': 'overall_dir'}

    drop_cols = ['record_no']

    # The directions are in an odd format, we will change them
    trend_arrow_map = {1: -1,
                       2: 0,
                       3: 1}

    def __init__(self, dataframe):
        super().__init__(dataframe)
        self.col_map.update(self.change_col_map)

    def transform(self):
        super().transform()
        self._map_directions()

    def _map_directions(self):
        """ Maps the trend direction columns. """

        for col in ('achievement_dir', 'growth_dir', 'overall_dir'):
            self.df[col] = self.df[col].map(self.trend_arrow_map)


class CoactMaker(KaggleMaker):
    # district_name is dropped because it is not present in all datasets
    drop_cols = ['district_name']

    # The college readiness used two different standards for true/false
    readiness_map = {1: 1,
                     2: 0,
                     0: 0}

    def transform(self):
        super().transform()
        self._map_readiness()

    def _map_readiness(self):
        """ Maps the readiness columns """
        for col in ('eng_yn', 'math_yn', 'read_yn', 'sci_yn'):
            self.df[col] = self.df[col].map(self.readiness_map)


class EnrollMaker(KaggleMaker):
    drop_cols = ['unnamed: 12', 'unnamed: 13', 'unnamed: 14']


class FinalMaker(KaggleMaker):
    drop_cols = ['emh_2lvl',
                 'LT100pnts',
                 'aec10',
                 'alternative_school',
                 'charter',
                 'charteroronline',
                 'ell_growth_grade',
                 'emh_2lvl',
                 'final_plan',
                 'highestgrade',
                 'initial_plan',
                 'lowestgrade',
                 'lt100pnts',
                 'notes',
                 'online',
                 'overall_ach_grade',
                 'overall_achievement',
                 'record_no',
                 'spf_ps_ell_grad_rate',
                 'rank']

    final_col_map = {'aec_10': 'alternative_school',
                     'initial_plantype': 'initial_plan',
                     'final_plantype': 'final_plan',
                     'rank_tot': 'rank',
                     'overall_ach_grade': 'overall_achievement',
                     'read_ach_grade': 'read_achievement',
                     'read_ach_grade': 'read_achievement',
                     'math_ach_grade': 'math_achievement',
                     'write_ach_grade': 'write_achievement',
                     'sci_ach_grade': 'science_achievement',
                     'overall_weighted_growth_grade': 'overall_weighted_growth',
                     'read_growth_grade': 'read_growth',
                     'math_growth_grade': 'math_growth',
                     'write_growth_grade': 'write_growth',
                     'spf_ps_ind_grad_rate': 'graduation_rate'}

    def __init__(self, dataframe):
        super().__init__(dataframe)
        self.col_map.update(self.final_col_map)


class FrlMaker(KaggleMaker):
    drop_cols = ['unnamed: 5', 'unnamed: 6', 'unnamed: 7']

    frl_col_map = {'% free and reduced': 'pct_fr'}

    def __init__(self, dataframe):
        super().__init__(dataframe)
        self.col_map.update(self.frl_col_map)

    def transform(self):
        super().transform()
        self._drop_last_two_rows()
        # Clean the pct_fr column of any percentage signs
        self._clean_pct_signs('pct_fr')

    def _drop_last_two_rows(self):
        """ Removes the last two rows """

        self.df = self.df.drop([len(self.df) - 2, len(self.df) - 1])


class RemediationMaker(KaggleMaker):
    # district_names are dropped because they are written poorly
    drop_cols = ['unnamed: 5',
                 'this is 2011 data: created nov 7, 2012',
                 'public_private',
                 'district_name']

    rem_col_map = {'remediation_atleastone_pct2010': 'pct_remediation',
                   'remediation_at_leastone_pct2010': 'pct_remediation'}

    def __init__(self, dataframe):
        super().__init__(dataframe)
        self.col_map.update(self.rem_col_map)

    def transform(self):
        super().transform()
        # Clean the pct_remediation column from any percentage signs
        self._clean_pct_signs('pct_remediation')


class AddressMaker(KaggleMaker):
    drop_cols = ['phone', 'physical address']
    address_col_map = {'physical city': 'city',
                       'physical state': 'state',
                       'physical zipcode': 'zipcode'}

    def __init__(self, dataframe):
        super().__init__(dataframe)
        self.col_map.update(self.address_col_map)


class GPSMaker(Maker):
    drop_cols = ['school name']
    col_map = {'school number': 'school_id',
               'lattitude': 'latitude'}
