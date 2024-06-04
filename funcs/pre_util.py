#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilitary files to preprocess large sampling data assiciated with passive acoustic 
monitoring


"""
import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from os import listdir
from maad import sound, util
import matplotlib.pyplot as plt
import seaborn as sns

def search_files(directory=".", extension=""):
    extension = extension.lower()
    for dirpath, dirnames, files in os.walk(directory):
        for name in files:
            if extension and name.lower().endswith(extension):
                return os.path.join(dirpath, name)
            elif not extension:
                return os.path.join(dirpath, name)


def listdir_pattern(path_dir, ends_with=None):
    """
    Wraper function from os.listdir to include a filter to search for patterns
    
    Parameters
    ----------
        path_dir: str
            path to directory
        ends_with: str
            pattern to search for at the end of the filename
    Returns
    -------
    """
    flist = listdir(path_dir)

    new_list = []
    for names in flist:
        if names.endswith(ends_with):
            new_list.append(names)
    return new_list

def add_file_prefix(folder_name: str, recursive:bool=False) -> None:
    """
    Adds a prefix to the file names in the given directory.
    The prefix is the name of the immediate parent folder of the files.

    Parameters:
    folder_name(str): Name of directory which contains files.
    recursive(bool): If True, searches for files in sub-directories recursively.
                     Defaults to False if not provided.

    Returns: None
    """
    folder_path = Path(folder_name)

    # Get list of files to process
    if recursive:
        flist = list(folder_path.glob('**/*.WAV')) + list(folder_path.glob('**/*.wav'))
    else:
        flist = list(folder_path.glob('*.WAV')) + list(folder_path.glob('*.wav'))

    flist = [f for f in flist if (not f.name.startswith(f.parent.name+'_'))]
    flist = [f for f in folder_path.glob('[!.]*.WAV')] + [f for f in folder_path.glob('[!.]*.wav')]

    print(f'Number of WAVE files detected with no prefix: {len(flist)}')

    # Loop and change names
    for fname in flist:
        prefix = fname.parent.name
        new_fname = fname.with_name(f'{prefix}_{fname.name}')
        try:
            fname.rename(new_fname)
        except Exception as e:
            print(f"Error occurred while renaming {fname}: {e}")

def metadata_summary(df):
    """ Get a summary of a metadata dataframe of the acoustic sampling

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe must have the columns site, date, sample_length.
        Use maad.util.get_audio_metadata to compile the dataframe.

    Returns
    -------
    pandas DataFrame
        A summary of each site
    """
    df.loc[:,'date_fmt'] = pd.to_datetime(df.date,  format='%Y-%m-%d %H:%M:%S')
    df_summary = {}
    for i, df_site in df.groupby('sensor_name'):
        site_summary = {
            'date_ini': str(df_site.date_fmt.min()),
            'date_end': str(df_site.date_fmt.max()),
            'n_recordings': len(df_site),
            'duration': str(df_site.date_fmt.max() - df_site.date_fmt.min()),
            'sample_length': df_site.length.mean().round(),
            'time_diff': df_site['date_fmt'].sort_values().diff().median(),
        }
        df_summary[i] = site_summary
    df_summary = pd.DataFrame(df_summary).T
    df_summary.reset_index().rename(columns={'index': 'sensor_name'})
    return df_summary

def plot_sensor_deployment(df, ax=None):
    """ Plot sensor deployment to have an overview of the sampling

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe must have the columns site, date, sample_length.
        Use maad.util.get_audio_metadata to compile the dataframe.
    ax : matplotlib.axes, optional
        Matplotlib axes fot the figure, by default None

    Returns
    -------
    matplotlib.figure
        If axes are not provided, a figure is created and figure handles are returned.
    """
    df.loc[:,'date_fmt'] = pd.to_datetime(df.date,  format='%Y-%m-%d %H:%M:%S')
    df_out = pd.DataFrame()
    for sensor_name, df_sensor in df.groupby('sensor_name'):
        aux = pd.DataFrame(df_sensor.date_fmt.dt.date.value_counts())
        aux['sensor_name'] = sensor_name
        aux.reset_index(inplace=True)
        aux.rename(columns={'index': 'date', 'date_fmt': 'num_rec'}, inplace=True)
        df_out = pd.concat([df_out, aux], axis=0)

    df_site = pd.DataFrame()
    for sensor_name, df_sensor in df.groupby('sensor_name'):
        df_sensor.sort_values('date', inplace=True)
        site_info = pd.DataFrame({
            'sensor_name': sensor_name,
            'sample_rate': int(df_sensor.sample_rate.iloc[0]),
            'bit_depth': int(df_sensor.bits.iloc[0]),
            'date_ini': df_sensor.date.iloc[0][0:10],
            'date_end': df_sensor.date.iloc[-1][0:10],
            'time_ini': df_sensor.date.iloc[0][11:19],
            'time_end': df_sensor.date.iloc[-1][11:19]
            }, index=[sensor_name])
        df_site = pd.concat([df_site, site_info], ignore_index=True)


        sns.scatterplot(y='num_rec', x='sensor_name', 
                        size='count', size_norm = (10, 100), 
                        hue='count', hue_norm = (10, 100),
                        data=df_out)


def random_sample_metadata(df, n_samples_per_site=10, hour_sel=None, random_state=None):
    """ Get a random sample form metadata DataFrame """
    if hour_sel==None:
        hour_sel = [str(i).zfill(2) for i in range(24)]

    # format data
    df.loc[:,'hour'] = df.loc[:,'date'].str[11:13].astype(str)
    sensor_list = df.sensor_name.unique()

    # Batch process
    df_out = pd.DataFrame()
    for sensor_name in sensor_list:
        df_sel = df.loc[(df.sensor_name==sensor_name) & (df.hour.isin(hour_sel)),:]
        df_sel = df_sel.sample(n_samples_per_site, random_state=random_state)
        df_out = pd.concat([df_out, df_sel])

    df_out.reset_index(drop=True, inplace=True)

    # Check that all sites have 6 samples
    # df_out.sensor_name.value_counts()
    # Check that times are distributed in a uniform way
    # df_out.hour.value_counts() / len(df_out)

    return df_out

def copy_file_list(flist, path_save):
    #%% Copy selected files to a new folder
    for _, row in flist.iterrows():
        src_file = row.path_audio
        dst_file = path_save + row.fname
        shutil.copyfile(src_file, dst_file)

def concat_audio(flist, sample_len=1, display=False):
    """ Concatenates samples using a list of audio files

    Parameters
    ----------
    flist : list or pandas Series
        List of files to concatenate
    sample_len : float, optional
        Length in seconds of each sample, default is 1 second
    display : bool, optional
        If true displays the audio spectrogram, by default False
    
    Return
    ------
    """
    
    # Compute long wav
    long_wav = list()
    for idx, fname in enumerate(flist):
        s, fs = sound.load(fname)
        s = sound.trim(s, fs, 0, sample_len)
        long_wav.append(s)

    long_wav = np.concatenate(long_wav)
    
    # Plot
    if display:
        Sxx, tn, fn, ext = sound.spectrogram(
            long_wav, fs, window='hann', nperseg=1024, noverlap=512)
        fig, ax = plt.subplots(1,1, figsize=(10,3))
        util.plot_spectrogram(Sxx, extent=[0, 24, 0, 11],
                            ax=ax, db_range=80, gain=25, colorbar=False)
        ax.set_xlabel('Time [Hours]')
        ax.set_xticks(range(0,25,4))

    return long_wav, fs