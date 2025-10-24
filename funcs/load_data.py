# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 11:07:09 2023

@author: Esteban
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class load_data:
    def __init__(self):
        pass

    @staticmethod
    def find(s, ch):
        return [i for i, ltr in enumerate(s) if ltr == ch]

    @classmethod
    def create_df(cls, path, flist, recorder, fload):
        d=pd.DataFrame()
        L=str(len(flist))
        for i,j in enumerate(flist):
            if fload=='file':
                file = os.path.basename(flist)
            else:
                file = os.path.basename(j)
                
            if recorder=='SongMeter: name_aammdd_hhmmss':
                ind=cls.find(file,'_')
                ind.append(file.find('.'))
                d.loc[i,'route']=flist[i]; d.loc[i,'file']=file; d.loc[i,'site']=file[0:ind[0]]
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
                d.loc[i,'route']=flist[i]; d.loc[i,'file']=file; d.loc[i,'site']=site
                d.loc[i,'date_fmt']=pd.to_datetime(file[0:ind[0]]+' '+file[ind[0]+1:ind[1]], format='%Y%m%d %H%M%S')
                d.loc[i,'day']=file[0:ind[0]]
                d.loc[i,'hour']=d.loc[i,'date_fmt'].hour+d.loc[i,'date_fmt'].minute/60
                
            elif recorder=='Snap: name_aammddThhmmss':
                ind_=cls.find(file,'_')
                indT=cls.find(file,'T')
                indp=cls.find(file,'.')
                d.loc[i,'route']=flist[i]; d.loc[i,'file']=file; d.loc[i,'site']=file[ind_[0]+1:ind_[1]]
                d.loc[i,'date_fmt']=pd.to_datetime(file[ind_[1]+1:indT[0]]+' '+file[indT[0]+1:indp[0]], format='%Y%m%d %H%M%S')
                d.loc[i,'day']=file[ind_[1]+1:indT[0]]
                d.loc[i,'hour']=d.loc[i,'date_fmt'].hour+d.loc[i,'date_fmt'].minute/60

            if fload == 'file': # Break for cycle so flist is assigned to route in the first iteration. This is only necessary for files
                d.loc[0,'route']=flist
                break
            
            print('progress: ' + str(i+1) +' of ' + L)    
        return(d)  

    @classmethod
    def plot_folder_stats(cls, df):
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
        indsep = cls.find(txt, ':')
        init = pd.to_datetime(txt[0:indsep[0]], format='%Y%m%d')
        fin = pd.to_datetime(txt[indsep[0]+1:len(txt)], format='%Y%m%d')
        # boolean indexing to avoid pandas query with @ variables
        df_day = df[(df['date_fmt'] >= init) & (df['date_fmt'] < fin)]
        return df_day

# Backward compatibility: allow calling as module-level functions
create_df = load_data.create_df
plot_folder_stats = load_data.plot_folder_stats
plot_set_recorders = load_data.plot_set_recorders
update_df = load_data.update_df