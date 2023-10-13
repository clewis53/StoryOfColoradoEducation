# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 20:29:54 2023

@author: caeley
"""
from pathlib import Path

def append_path(path, addition):
    """
    Appends the path to a string based on type.

    Parameters
    ----------
    path : str, Path
        The original path
    addition : String
        The string to append to the path
        
    Raises
    ------
    TypeError
        The file_extension must contain {year}.

    Returns
    -------
    str, Path
        The original path with the addition

    """
    if type(addition) != str:
        raise TypeError('The addition must be of type string')
    
    if type(path) == str:
        return path + '/' + addition
    elif Path in type(path).mro():
        return path.joinpath(addition)
    else:
        raise TypeError('Path must be of type string or Path')
        

def create_filenames(filepath, file_extension='{year}', years=(2010,2011,2012)):
    """
    A function to append a year file_extension to each final path

    Parameters
    ----------
    filepath : String, Path
        The filepath
    file_extension : String, optional
        The extension to add to each file location. Must_containe {years}.
        The default is '{year}'.
    years : , optional
        The years to add. The default is (2010,2011,2012).

    Raises
    ------
    ValueError
        The file_extension must contain {year}.

    Returns
    -------
    list
        A list of filepaths.

    """
    if '{year}' not in file_extension:
        raise ValueError('{year} must be in the file_extension')
    
    return [append_path(filepath, file_extension.format(year=year)) for year in years]
