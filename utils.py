# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 11:07:09 2023

@author: Esteban
"""
import os
from tkinter.filedialog import askdirectory
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from resampy import resample
from scipy.io import wavfile


class Loader:
    def __init__(self):
        pass

    @staticmethod
    def find(s, ch):
        """Find all indices of character 'ch' in string 's'."""
        return [i for i, ltr in enumerate(s) if ltr == ch]

    @classmethod
    def create_df(cls, path, flist, recorder, fload):
        """Create a DataFrame with metadata extracted from audio file names."""
        d=pd.DataFrame()
        L=str(len(flist))
        for i,j in enumerate(flist):
            if fload=='file':
                file = os.path.basename(flist)
            else:
                file = os.path.basename(j)
                
            if recorder=='Grillo: nombre_aammdd_hhmmss':
                ind=cls.find(file,'_')
                ind.append(file.find('.'))
                d.loc[i,'route']=flist[i]
                d.loc[i,'file']=file
                d.loc[i,'site']=file[0:ind[0]]
                d.loc[i,'date_fmt']=pd.to_datetime(file[ind[0]+1:ind[1]]+' '+file[ind[1]+1:ind[2]], format='%Y%m%d %H%M%S')
                d.loc[i,'day']=file[ind[0]+1:ind[1]]
                d.loc[i,'hour']=d.loc[i,'date_fmt'].hour+d.loc[i,'date_fmt'].minute/60

            elif recorder=='Audiomoth: aammdd_hhmmss':
                site = ''
                if fload=='file' or fload == 'folder':
                    # Use folder name as site
                    site = os.path.basename(path)
                elif fload == 'set':
                    # Parent folder of the file is the site
                    site = os.path.basename(os.path.dirname(j))
                    
                ind=cls.find(file,'_')
                ind.append(file.find('.'))
                d.loc[i,'route']=flist[i]
                d.loc[i,'file']=file
                d.loc[i,'site']=site
                d.loc[i,'date_fmt']=pd.to_datetime(file[0:ind[0]]+' '+file[ind[0]+1:ind[1]], format='%Y%m%d %H%M%S')
                d.loc[i,'day']=file[0:ind[0]]
                d.loc[i,'hour']=d.loc[i,'date_fmt'].hour+d.loc[i,'date_fmt'].minute/60

            if fload == 'file': # Break for cycle so flist is assigned to route in the first iteration. This is only necessary for files
                d.loc[0,'route']=flist
                break
            
            print('progress: ' + str(i+1) +' of ' + L)    
        return(d)  

    @classmethod
    def plot_folder_stats(cls, df):
        """Plot the distribution of recordings by day and hour."""
        days, indd = np.unique(df.day, return_inverse=True)   
        hour=df.hour
        hours, indh = np.unique(hour, return_inverse=True)   

        countd=np.unique(indd, return_counts=True)
        counth=np.unique(indh, return_counts=True)
        
        plt.figure(figsize=(10,6))
        plt.subplot(1,2,1)
        plt.barh(days, countd[1], align='center', alpha=0.5)
        plt.ylabel('Day')
        plt.xlabel('File count')
        plt.subplot(1,2,2)
        plt.barh(hours, counth[1], align='center', alpha=0.5)
        plt.ylabel('Hour')
        plt.xlabel('File count')
        plt.show()
        
    @classmethod
    def plot_set_recorders(cls, df):
        """Plot the distribution of recordings by site and day."""
        d = df.groupby('site')
        dc = pd.DataFrame(d.day.value_counts())
        dc.rename(columns={'day': 'count'}, inplace=True)
        dc.sort_values(by='day', ascending=True, inplace=True)
        sns.scatterplot(y='day', x='site', size='count', hue='count',
                        size_norm=(10, 100), hue_norm=(10, 100), data=dc)
        plt.show()
        df_summary = {}
        for i, df_site in df.groupby('site'):
            site_summary = {
                'date_ini': str(df_site.date_fmt.min()),
                'date_end': str(df_site.date_fmt.max()),
                'n_recordings': len(df_site),
                'duration': str(df_site.date_fmt.max() - df_site.date_fmt.min()),
                'time_diff': df_site['date_fmt'].sort_values().diff().median(),
            }
            df_summary[i] = site_summary
        df_summary = pd.DataFrame(df_summary).T
        df_new = df_summary.reset_index().rename(columns={'index': 'site'})
        return df_new

    @classmethod
    def update_df(cls, df, txt):
        """Update the DataFrame to include only recordings within a specified date range."""
        indsep = cls.find(txt, ':')
        init = pd.to_datetime(txt[0:indsep[0]], format='%Y%m%d')
        fin = pd.to_datetime(txt[indsep[0]+1:len(txt)], format='%Y%m%d')
        # boolean indexing to avoid pandas query with @ variables
        df_day = df[(df['date_fmt'] >= init) & (df['date_fmt'] < fin)]
        return df_day
    
    @classmethod
    def show_dataframe(cls, df_to_show: pd.DataFrame, title: str = "DataFrame",root_window=None):
        """Open a new Tkinter window with a scrollable table (ttk.Treeview) to display a DataFrame.

        - Shows columns as headers
        - Adds vertical and horizontal scrollbars
        - Adapts column widths based on header length
        """
        if df_to_show is None or (isinstance(df_to_show, pd.DataFrame) and df_to_show.empty):
            print('No hay metadatos para mostrar')
            return

        top = tk.Toplevel(root_window.window if root_window else None)
        top.title(title)
        top.geometry("900x500")

        container = ttk.Frame(top)
        container.pack(fill='both', expand=True)

        # Create Treeview
        cols = list(df_to_show.columns)
        tree = ttk.Treeview(container, columns=cols, show='headings')

        # Scrollbars
        vsb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Grid placement
        tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Configure headings and column widths
        for c in cols:
            tree.heading(c, text=str(c))
            # Set a reasonable default width based on header length
            width = max(100, min(300, 10 * len(str(c))))
            tree.column(c, width=width, stretch=True, anchor='w')

        # Insert data rows
        for _, row in df_to_show.iterrows():
            values = [row[c] for c in cols]
            # Convert complex objects to string for display
            values = [str(v) if not (isinstance(v, (int, float, str)) or pd.isna(v)) else ("" if pd.isna(v) else v) for v in values]
            tree.insert('', 'end', values=values)
    
class Sampler:
    """Class for resampling audio files in the dataset."""
    def __init__(self):
        pass

    def resample_dataset(self,df,ftarget=48000):
        """Resample audio files in the dataset to a target sampling frequency."""
        save_dir=askdirectory(title="Seleccione carpeta para guardar los archivos resampleados")   
        print('Resampleando archivos...')
        for i, row in df.iterrows():
            p=Path(row.route)
            folder=p.parent.name
            save_path = os.path.join(save_dir, folder)
            filename= save_dir + '/' + str(folder) + '/' + str(p.name)
            if os.path.isfile(filename):
                print("Archivo ya existe")  
            else:
            
                try :
                    fs, s = wavfile.read(row.route)
                    print('Archivo cargado: ' + str(row.route))
                except ValueError:
                    print('Error resampling file: ' + str(row.route))

                if fs != ftarget:
                    s_res = resample(s, fs, ftarget, res_type='kaiser_fast')
                else:
                    s_res = s

                try:
                    # Create the directory
                    os.mkdir(save_path)
                    print(f"Directory created: {save_path}")
                except FileExistsError:
                    print("Carpeta ya existe")

                wavfile.write(filename, ftarget, s_res.astype(np.int16)) # Save the resampled audio file format as 16-bit PCM
                print('Archivo creado:')
                print(filename)

            print('progreso: ' + str(i) + ' de ' + str(len(df)))
        