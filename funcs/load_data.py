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
from maad import sound

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

class Loader:
    def __init__(self):
        pass
        
    def create_df(path,flist,recorder,fload):
        d=pd.DataFrame()
        L=str(len(flist))
        for i,j in enumerate(flist):
            if fload=='file':
                file = os.path.basename(flist)
            else:
                file = os.path.basename(j)
                
            if recorder=='SongMeter: name_aammdd_hhmmss':
                ind=find(file,'_')
                ind.append(file.find('.'))
                d.loc[i,'route']=flist[i]; d.loc[i,'file']=file; d.loc[i,'site']=file[0:ind[0]];            
                d.loc[i,'date_fmt']=pd.to_datetime(file[ind[0]+1:ind[1]]+' '+file[ind[1]+1:ind[2]], format='%Y%m%d %H%M%S')
                d.loc[i,'day']=file[ind[0]+1:ind[1]]
                d.loc[i,'hour']=d.loc[i,'date_fmt'].hour+d.loc[i,'date_fmt'].minute/60

            elif recorder=='Audiomoth: aammdd_hhmmss':
                if fload=='file' or fload == 'folder':
                    ind_site=find(path,'/')
                    site=path[ind_site[-1]+1:len(path)]     
                elif fload == 'set':
                    ind_site=find(j,'\\')  
                    site=j[ind_site[-2]+1:ind_site[-1]]
                    
                ind=find(file,'_')
                ind.append(file.find('.'))                
                d.loc[i,'route']=flist[i]; d.loc[i,'file']=file; d.loc[i,'site']=site;            
                d.loc[i,'date_fmt']=pd.to_datetime(file[0:ind[0]]+' '+file[ind[0]+1:ind[1]], format='%Y%m%d %H%M%S'),
                d.loc[i,'day']=file[0:ind[0]];
                d.loc[i,'hour']=d.loc[i,'date_fmt'].hour+d.loc[i,'date_fmt'].minute/60;
                
            elif recorder=='Snap: name_aammddThhmmss':
                ind_=find(file,'_')
                indT=find(file,'T')
                indp=find(file,'.')            
                d.loc[i,'route']=flist[i]; d.loc[i,'file']=file; d.loc[i,'site']=file[ind_[0]+1:ind_[1]];            
                d.loc[i,'date_fmt']=pd.to_datetime(file[ind_[1]+1:indT[0]]+' '+file[indT[0]+1:indp[0]], format='%Y%m%d %H%M%S'),
                d.loc[i,'day']=file[ind_[1]+1:indT[0]];
                d.loc[i,'hour']=d.loc[i,'date_fmt'].hour+d.loc[i,'date_fmt'].minute/60;

            if fload == 'file': #Break for cycle so flist is assigned to rout in the first iteration. This is only necesarry for files
                d.loc[0,'route']=flist
                break
            
            print('progress: ' + str(i+1) +' of ' + L)    
        return(d) 


    def plot_folder_stats(df):
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
        
    def plot_set_recorders(df):    
        d=df.groupby('site')
        dc=pd.DataFrame(d.day.value_counts())
        dc.rename(columns={'day': 'count'}, inplace=True)  
        dc.sort_values(by='day',ascending = True, inplace=True)
        sns.scatterplot(y='day', x='site', size='count', hue='count',
                        size_norm = (10, 100), hue_norm = (10, 100), data=dc)
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
        df_new=df_summary.reset_index().rename(columns={'index': 'site'})    
        return(df_new)

    def update_df(df,txt):
        indsep=find(txt,':')
        init=pd.to_datetime(txt[0:indsep[0]], format='%Y%m%d')
        fin=pd.to_datetime(txt[indsep[0]+1:len(txt)], format='%Y%m%d')        
        df_day = df.query("date_fmt >= @init & date_fmt < @fin")
        return(df_day)

    #no funcionó, muy lenta la función
    """
    def gen_spec_matrix(df,samp_len,flims,db,ch):   
        L=len(df)
        j=0
        for i,file_name in enumerate(df.route):
            s, fs = sound.load(df.route[df.index[j]])            
            if ch==1:
                s = sound.trim(s, fs, 0, samp_len)  
                s = sound.resample(s, fs, flims[1]*2+4000, res_type='kaiser_fast')
                fs=flims[1]*2+4000
                
            Sxx, tn, fn, _ = sound.spectrogram(s, fs, window='hann', nperseg=1024, noverlap=512,flims=flims)        
            
            if j==0:
                ii, jj=Sxx.shape; 
                Sxxx=np.zeros((ii,jj,L))
                #Sxxx_ind=np.zeros((1,L))
                
            sh1=list(Sxx.shape); sh2=list(Sxxx.shape)[0:2]   
                
            if sh1 == sh2: 
                Sxxx[:,:,j]=Sxx
                #Sxxx_ind[j]=df.index[j]
                
            print('progress: ' + str(j+1) +' of ' + str(L))  
            j=j+1
        return(Sxxx)
    """    