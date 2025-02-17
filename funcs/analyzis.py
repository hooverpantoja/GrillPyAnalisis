# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:50:30 2023

@author: Esteban
"""

import numpy as np
import pandas as pd
import seaborn as sns
from maad import sound, util, rois, features, spl
import matplotlib.pyplot as plt
from skimage.io import imshow
from skimage.transform import downscale_local_mean
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def longwave(df,path,samp_len,flims,db,ch):
    long_wav = list()
    L=str(len(df.file))
    for idx, fname in enumerate(df.file):   
         
         s, fs = sound.load(path + '/' + fname)         
         s = sound.trim(s, fs, 0, samp_len)
         if ch==1:
             s = sound.resample(s, fs, flims[1]*2+4000, res_type='kaiser_fast')
             fs=flims[1]*2+4000
         long_wav.append(s)             
         print('progress: ' + str(idx+1) +' of ' + L)    
    long_wav = util.crossfade_list(long_wav, fs, fade_len=0.5)
    Sxx, tn, fn, _ = sound.spectrogram(long_wav, fs, 
                                         window='hann', nperseg=1024, noverlap=512,flims=flims)           
    Sxx_db=20*np.log10(Sxx/Sxx.min())
    Sxx_db[Sxx_db < db]=db   
    fig, ax = plt.subplots()
    c = ax.pcolorfast(tn, fn/1000, Sxx_db, cmap='gray', vmin=Sxx_db.min(), vmax=Sxx_db.max())
    plt.ylabel('Frequency [kHz]'); plt.xlabel('Time (s)') 
    cbar=fig.colorbar(c, ax=ax)
    cbar.set_label('dB in ref to min')
    plt.show(block=False)                                           
    return(long_wav,fs,Sxx,tn,fn)

def shortwave(s,fs,flims,db):
    #target_fs=48000
    #s = sound.resample(s, fs, target_fs, res_type='kaiser_fast')
    Sxx, tn, fn, ext = sound.spectrogram(s, fs,                                          
                                         window='hann', nperseg=1024, noverlap=512,flims=flims)
    Sxx_db=20*np.log10(Sxx/Sxx.min())
    Sxx_db[Sxx_db < db]=db
    fig, ax = plt.subplots()
    c = ax.pcolorfast(tn, fn/1000, Sxx_db, cmap='gray', vmin=Sxx_db.min(), vmax=Sxx_db.max())
    plt.ylabel('Frequency [kHz]'); plt.xlabel('Time (s)') 
    cbar=fig.colorbar(c, ax=ax)
    cbar.set_label('dB in ref to min')
    plt.show(block=False)    
    
def rois_spec(data,flims,ch_rec,db,bstd,bp):   
    #target_fs=48000
    #s = sound.resample(s, fs, target_fs, res_type='kaiser_fast')
    s_filt=data.wav    
    Sxx, tn, fn, ext = sound.spectrogram(s_filt, data.wavfs, nperseg=1024, noverlap=512,flims=flims)                  
    Sxx_db=20*np.log(Sxx/Sxx.min())
    Sxx_db[Sxx_db < db] = 0
    #Sxx[Sxx_db < db]=0
    #Sxx_db_rmbg, _, _ = sound.remove_background(Sxx_db)
    Sxx_db= sound.smooth(Sxx_db,std=0.05)   
    Sxxf=Sxx_db/Sxx_db.max()    
    #im_mask = rois.create_mask(im=Sxxf, mode_bin ='relative', bin_std=bstd, bin_per=bp)
    im_mask = rois.create_mask(im=Sxxf, mode_bin ='absolute',bin_h=bstd, bin_l=bp)  
    im_rois, df_rois = rois.select_rois(im_mask, min_roi=1, max_roi=5000) 
    df_rois = util.format_features(df_rois, tn, fn)
    fig_kwargs = {'vmin':Sxxf.min(), 'vmax':Sxxf.max(),
                  'extent':(tn[0], tn[-1], fn[0]/1000, fn[-1]/1000),
                  'figsize': (4,13),
                  'title':'spectrogram',
                  'xlabel':'Time [s]',
                  'ylabel':'Frequency [kHz]',
                  }
    util.overlay_rois(Sxxf, df_rois, **fig_kwargs) 
    plt.show(block=False) 
    return df_rois, im_rois

def ind_batch(df):
    print('in progress')
    df.reset_index()
    L=str(len(df.route))
    for i,file_name in enumerate(df.route):            
        temp, f_ok =compute_acoustic_indices(file_name)             
        if f_ok == 'error':
            print('error calculating index')
        else:
            df.loc[df.index[i],'ADI']=temp.ADI
            df.loc[df.index[i],'ACI']=temp.ACI
            df.loc[df.index[i],'NDSI']=temp.NDSI
            df.loc[df.index[i],'BI']=temp.BI
            df.loc[df.index[i],'Hf']=temp.Hf
            df.loc[df.index[i],'Ht']=temp.Ht
            df.loc[df.index[i],'H']=temp.H
            df.loc[df.index[i],'SC']=temp.SC
            df.loc[df.index[i],'NP']=temp.NP

            print('progress: ' + str(i+1) +' of ' + L) 
    print('indices calculated')
    return(df)

def ind_per_day(df):
    print('in progress')    
    group_site=df.groupby(['site'])
    Ls=len(group_site)
    k=0
    ind_sites=list()
    for site, data in group_site:        
        print('site: ' + site+ ' # '+str(k+1) +' of ' + str(Ls) + ' sites') 
        group_day=data.groupby(['day'])
        Ld=len(group_day)
        i=0 
        #indx_d=np.zeros([Ld])  
        s_long=list()
        for day, rows in group_day:
            L=len(rows)
            j=0
            for indx, row in rows.iterrows():
                file=row['route']                 
                s, fs= sound.load(file)            
                s = sound.resample(s, fs, 48000, res_type='kaiser_fast')
                fs=48000 
                s_long.append(s)             
                j=j+1
                print(j)
            #Calculate indices
            S,fn = sound.spectrum(s_long, fs, window='hann', nperseg=1024, noverlap=512)        
       
            
            i=i+1
            print('progress: ' + str(i) +' of ' + str(Ld)) 
        ind_sites.append(site) 
        k=k+1         
        print('progress: ' + str(k) +' of ' + str(Ls)) 
    
    
    df.reset_index()
    L=str(len(df.route))
    for i,file_name in enumerate(df.route):            
        temp, f_ok =compute_acoustic_indices(file_name)             
        if f_ok == 'error':
            print('error calculating index')
        else:
            df.loc[df.index[i],'ADI']=temp.ADI
            df.loc[df.index[i],'ACI']=temp.ACI
            df.loc[df.index[i],'NDSI']=temp.NDSI
            df.loc[df.index[i],'BI']=temp.BI
            df.loc[df.index[i],'Hf']=temp.Hf
            df.loc[df.index[i],'Ht']=temp.Ht
            df.loc[df.index[i],'H']=temp.H
            df.loc[df.index[i],'SC']=temp.SC
            df.loc[df.index[i],'NP']=temp.NP

            print('progress: ' + str(i+1) +' of ' + L) 
    print('indices calculated')
    return(df)

def saturation(S,fn,tn):
    S=S[fn>2000]
    #f=fn[fn>2000]
    values, counts = np.unique(S, return_counts=True)
    mc=values[counts.argmax()]  
    Sxx_db=20*np.log(S/mc)  
   # q2=np.quantile(Sxx_db,0.5)
    Sxx_b=Sxx_db > 100
   # fig, ax = plt.subplots()
   # ax.pcolorfast(tn, f/1000, Sxx_b, cmap='gray', vmin=Sxx_b.min(), vmax=Sxx_b.max())
   # plt.ylabel('Frequency [kHz]'); plt.xlabel('Time (s)') 
   # plt.show(block=False)
    x,y=Sxx_b.shape
    sat=Sxx_b.sum()/(x*y)
    return(sat)
    
def compute_acoustic_indices(path_audio):
    """
    Parameters
    ----------
    s : 1d numpy array
        acoustic data
    Sxx : 2d numpy array of floats
        Amplitude spectrogram computed with maad.sound.spectrogram mode='amplitude'
    tn : 1d ndarray of floats
        time vector with temporal indices of spectrogram.
    fn : 1d ndarray of floats
        frequency vector with temporal indices of spectrogram..

    Returns
    -------
    df_indices : pd.DataFrame
    """
    s, fs = sound.load(path_audio)
    if 's' in locals():
        target_fs=48000
        s = sound.resample(s, fs, target_fs, res_type='kaiser_fast')
    
        # Compute the amplitude spectrogram and acoustic indices
        Sxx, tn, fn, ext = sound.spectrogram(
            s, target_fs, nperseg = 1024, noverlap=0, mode='amplitude')
        
        # Set spectro as power (PSD) and dB scales.
        Sxx_power = Sxx**2
        Sxx_dB = util.amplitude2dB(Sxx)
    
        # Compute acoustic indices
        ADI = features.acoustic_diversity_index(Sxx, fn, fmin=2000, fmax=24000, bin_step=1000, index='shannon', dB_threshold=-70)
        _, _, ACI = features.acoustic_complexity_index(Sxx)
        NDSI, xBA, xA, xB = features.soundscape_index(Sxx_power, fn, flim_bioPh=(2000, 20000), flim_antroPh=(0, 2000))
        Ht = features.temporal_entropy(s)
        Hf, _ = features.frequency_entropy(Sxx_power)
        H = Hf * Ht
        BI = features.bioacoustics_index(Sxx, fn, flim=(2000, 11000))
        NP = features.number_of_peaks(Sxx_power, fn, mode='linear', min_peak_val=0, 
                                      min_freq_dist=100, slopes=None, prominence=1e-6)
        #_, SC, _ = features.spectral_cover(Sxx_dB, fn, dB_threshold=-50, flim_MF=(4000,20000))
        SC=saturation(Sxx,fn,tn)
        df_indices = pd.Series({
            'ADI': ADI,
            'ACI': ACI,
            'NDSI': NDSI,
            'BI': BI,
            'Hf': Hf,
            'Ht': Ht,
            'H': H,
            'NP': int(NP),
            'SC': SC})
            
        flag='ok'
        return df_indices,flag
    else:  
        df_indices=1; flag='error'
        return df_indices,flag


def plot_acoustic_indices(df):
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    sns.scatterplot(data=df, x='hour', y='ACI', ax=ax[0,0],hue='site')
    sns.scatterplot(data=df, x='hour', y='ADI', ax=ax[0,1],hue='site')
    sns.scatterplot(data=df, x='hour', y='BI', ax=ax[0,2],hue='site')
    sns.scatterplot(data=df, x='hour', y='H', ax=ax[1,0],hue='site')
    sns.scatterplot(data=df, x='hour', y='Ht', ax=ax[1,1],hue='site')
    sns.scatterplot(data=df, x='hour', y='Hf', ax=ax[1,2],hue='site')
    sns.scatterplot(data=df, x='hour', y='NDSI', ax=ax[2,0],hue='site')
    sns.scatterplot(data=df, x='hour', y='NP', ax=ax[2,1],hue='site')
    sns.scatterplot(data=df, x='hour', y='SC', ax=ax[2,2],hue='site')
    fig.set_tight_layout('tight')
    plt.show()
    
def spl_batch(df):    
    S = -18         # Sensibility of the microphone -35dBV (SM4) / -18dBV (Audiomoth)
    G = 15       # Total amplification gain in dB (medium gain is 15 in audiomoth)
    VADC = 3.3  #VADC for audiomoth
    df.reset_index()
    L=str(len(df.file))
    for i,file_name in enumerate(df.route):
        w, fs= sound.load(file_name)  
        #spl in time domain
        tmp=spl.wav2dBSPL(wave=w, gain=G, Vadc=VADC, sensitivity=S)
        df.loc[df.index[i],'spl']=tmp.mean()
        #spl in frequency domain
        Sxx_tmp,tn,fn,ext = sound.spectrogram (w, fs, window='hann', nperseg = 1024, noverlap=1024//2,verbose = False, display = False,savefig = None)
        mean_PSD = np.mean(Sxx_tmp, axis = 1)
        spl_ant_tmp=[spl.psd2leq(mean_PSD[util.index_bw(fn,(0,3000))],gain=G,sensitivity=S,Vadc=VADC)]
        df.loc[df.index[i],'spl_ant']=spl_ant_tmp
        spl_bio_tmp=[spl.psd2leq(mean_PSD[util.index_bw(fn,(3000,24000))], gain=G, sensitivity=S, Vadc=VADC)]
        df.loc[df.index[i],'spl_bio']=spl_bio_tmp
        print('progress: ' + str(i+1) +' of ' + L) 
        
    df_summary = {}
    for i, df_site in df.groupby('site'):
        night = df_site.query("hour>= 0 & hour <= 4.5 | hour>= 19.5 & hour <= 24")    
        day = df_site.query("hour>= 6.5 & hour <= 17.5")            
        
        site_summary = {
            'site': df_site.site[df_site.index[0]],
            'date_ini': str(df_site.date_fmt.min()),
            'date_end': str(df_site.date_fmt.max()),
            
            'spl_mean_night': str(night.spl.mean()),
            'spl_mean_day': str(day.spl.mean()),
            'spl_std_night': str(night.spl.std()), 
            'spl_std_day': str(day.spl.std()), 
            
            'spl_ant_mean_night': str(night.spl_ant.mean()),
            'spl_ant_mean_day': str(day.spl_ant.mean()),
            'spl_ant_std_night': str(night.spl_ant.std()), 
            'spl_ant_std_day': str(day.spl_ant.std()), 
            
            'spl_bio_mean_night': str(night.spl_bio.mean()),
            'spl_bio_mean_day': str(day.spl_bio.mean()),
            'spl_bio_std_night': str(night.spl_bio.std()), 
            'spl_bio_std_day': str(day.spl_bio.std()), 
        }
        df_summary[i] = site_summary
    df_summary = pd.DataFrame(df_summary).T
    
    return(df,df_summary)       
        
def plot_spl(df):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    sns.scatterplot(data=df, x='hour', y='spl', hue='day')   
    fig.set_tight_layout('tight')
    plt.show()    
            
       
def ac_print(df):
    group_site=df.groupby(['site'])
    Ls=len(group_site)
    k=0
    Ssites=np.zeros([65,48,Ls])
    X=np.zeros([Ls,65*48])
    sites_ind=list()
    for site, data in group_site:        
        print('site: ' + site+ ' # '+str(k+1) +' of ' + str(Ls) + ' sites') 
        group_hour=data.groupby(['hour'])
        Lh=len(group_hour)
        i=0 
        Stt=np.zeros([513,Lh])
        tn=np.zeros([Lh])
        for hour, rows in group_hour:
            tn[i]=hour
            L=len(rows)
            St=np.zeros(513)   
            j=0
            for indx, row in rows.iterrows():
                file=row['route']
                s, fs= sound.load(file)            
                s = sound.resample(s, fs, 48000, res_type='kaiser_fast')
                fs=48000 
                Stmp,fn = sound.spectrum(s, fs, window='hann', nperseg=1024, noverlap=512)       
                St=St+Stmp
                j=j+1
                print(j)
            St=St/L
            Stt[:,i]=St            
            i=i+1
            print('progress: ' + str(i) +' of ' + str(Lh)) 
        values, counts = np.unique(Stt[128:384,:], return_counts=True)
        mc=values[counts.argmax()]  
        Stt_db=20*np.log10(Stt/mc)  
        mask=Stt_db>0     
        Sm = Stt_db * mask
        Sd = downscale_local_mean(Sm, (8, 1))
        plt.ylabel('Hour')
        plt.xlabel('Frequency (kHz)')
        plt.title(site)
        imshow(Sd, aspect='auto',origin='lower',extent=[tn[0],tn[len(tn)-1], 0 ,fn[-1]/1000]) 
        Ssites[:,:,k]=Sd
        X[k,:]=np.ravel(Sd, order = 'C')
        sites_ind.append(site)  
        plt.show(block=False)
        k=k+1         
        print('progress: ' + str(k) +' of ' + str(Ls)) 

    #Calculate NMDS
    dist_euclid = euclidean_distances(X)    
    metric=True
    dist_matrix=dist_euclid 
    y=np.transpose(sites_ind)
    
    mds = MDS(metric=metric, dissimilarity='precomputed', random_state=0)
    # Get the embeddings
    pts = mds.fit_transform(dist_matrix)
    # Plot the embedding, colored according to the class of the points
    fig = plt.figure(2, (15,6))
    ax = fig.add_subplot(1,2,1)    
    ax = sns.scatterplot(x=pts[:, 0], y=pts[:, 1], hue=y, palette=['r', 'g', 'b', 'c'])
    plt.title('Metric MDS with Euclidean')   
    plt.ylabel('NMDSy')
    plt.xlabel('NMDSx')
    # Add the second plot
    ax = fig.add_subplot(1,2,2)
    # Plot the points again
    plt.scatter(pts[:, 0], pts[:, 1])
    
    # Annotate each point by its corresponding face image
    for x, ind in zip(X, range(pts.shape[0])):
        im = x.reshape(65,48)
        imagebox = OffsetImage(im, zoom=0.7, cmap=plt.cm.viridis)
        i = pts[ind, 0]
        j = pts[ind, 1]
        ab = AnnotationBbox(imagebox, (i, j), frameon=False)
        ax.add_artist(ab)
    plt.title('Metric MDS with Euclidean')   
    plt.ylabel('NMDSy')
    plt.xlabel('NMDSx')
    plt.show(block=False)        
    return(X,y,pts)


        
        
        
        
        
        
        
        