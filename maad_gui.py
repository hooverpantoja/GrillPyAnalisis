# -*- coding: utf-8 -*-<
"""
Created on Fri Apr 14 10:56:10 2023

@author: Esteban

Read installation guide and readme file before execution 
"""
from funcs import load_data
from funcs import analyzis
import obj
import os
import tkinter as tk
from tkinter import ttk
import glob
import numpy as np
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
from maad import sound #util
import pandas as pd
# Init
df=obj.metadata(pd.DataFrame(),'_','_','_')
w=obj.widget('ch_rec','flims','txt','tlen','db','bstd','bper')
p=obj.path('_','_')
data=obj.data('wav','wavfs',pd.DataFrame(),pd.DataFrame())
flag=obj.flag('enter','load')
S=obj.spec('_','_','_','_')

###########################################
### Load data #############################
###########################################

### File #######################
def read_file():
    flag.load='file'    
    flist=askopenfilename()
    p.load=os.path.dirname(flist)
    os.path.dirname(flist)
    rec_format=cbox.get()
    df.md=load_data.create_df(p.load,flist,rec_format,flag.load)
    data.wav,data.wavfs= sound.load(flist)  
    if df.md.empty:
        print('Verify file and recorder type')
    else: 
        print('File loaded succesfully: ')
        print(flist)
    
### One recorder - One Folder #######################    
def get_data_files():
    flag.load='folder'
    p.load=askdirectory()
    flist=glob.glob(p.load +'/*.wav') #Reads *.wav files in the selected directory
    flist.sort()
    rec_format=cbox.get()
    df.md=load_data.create_df(p.load,flist,rec_format,flag.load)
    if df.md.empty:
        print('Verify folder and recorder type')
    else: 
        print('Folder loaded and dataframe created')
        _ ,fs = sound.load(flist[0]) 
        df.fs=fs
        load_data.plot_folder_stats(df.md)
        
### Set of recorders - Folder containing folders #######################   
def prep_recs():
    flag.load='set'
    p.load=askdirectory()
    flist=glob.glob(p.load +'/*/*.wav') #Reads *.wav files in the selected directory
    #flist.sort()
    rec_format=cbox.get()
    df.md=load_data.create_df(p.load,flist,rec_format,flag.load) 
    if df.md.empty:
        print('Verify folder and recorder type')
    else: 
        data.csv_summary=load_data.plot_set_recorders(df.md)
        print('Folder loaded and dataframe created') 
        print(data.csv_summary)
        
def sel_days():
    df.days, indices = np.unique(df.md.day, return_inverse=True)  
    print('Days to analyze: ')
    print(df.days)
    print('Write the days you want to analyze in the next line separated by ":" ex:20230504:20230507 and press enter:')
    w.txt=input()
    df_new=load_data.update_df(df.md, w.txt)
    if df_new.empty:        
        print('Error')
    else:
        print('Days selected and dataframe updated')
        df.md=[]; df.md=df_new
                
###########################################
### Functions #############################
###########################################

### For one file ##########################
def rois_gui():
    ch=ch_rec.get()
    if {flag.load=='file' or flag.load=='folder'} and ch==1:
        if [data.wav] != 0:
            w.flims, _ , w.db, w.bstd , w.bper = obj.get_info_widgets(ch_rec,fmine,
                                                               fmaxe,'_',data.wavfs,db,bstd,bp)
            #analyzis.rois_spec(data,w.flims,ch_rec,w.db)
            print('rois init')
            rois, im_rois=analyzis.rois_spec(data,w.flims,ch_rec,w.db,w.bstd,w.bper) 
            
        else:
            print('Load audio')
    elif flag.load=='set' and ch==1:
        print('aqui voy') 
    else:
        print('Load one file, one folder recorder, or a set of recorders. Also this analysis requires custom variables')


### For one Recorder ####################################                        
def one_day_spec():
    if flag.load=='file':
        w.flims, _ ,w.db, _ , _ , _ = obj.get_info_widgets(ch_rec,fmine,
                                                           fmaxe,'_',data.wavfs,db,'_','_')
        analyzis.shortwave(data.wav,data.wavfs,w.flims,w.db) 
    
    elif flag.load=='folder':
        w.flims,w.samp_len,w.db, _ , _,ch =obj.get_info_widgets(ch_rec,fmine,fmaxe,tlene,df.fs,db,'_','_')
        
        data.wav,data.wavfs,S.Sxx,S.tn,S.fn=analyzis.longwave(df.md,p.load,w.samp_len,w.flims,w.db,ch)
        print('Plot finished')
        
    elif flag.load=='set':
        print('Spectrogram can not be ploted for set of recorders')
    
### For one recorder or set of recorders #############       
"""def calculate_ind():         
    print('Calculating indices')       
    df_ind=analyzis.ind_batch(df.md)
    analyzis.plot_acoustic_indices(df_ind)  
    df.md=[]; df.md=df_ind
    data.csv_data=df_ind    
    print('Acoustic indices calculated')"""
    
def calculate_ind():         
    print('Calculating indices')       
    X=analyzis.ind_per_day(df.md)
    print(str(X))
        
def calculate_spl():     
    print('Calculating SPL')
    df_spl, df_sum=analyzis.spl_batch(df.md)    
    analyzis.plot_spl(df_spl) 
    df.md=[]; df.md=df_spl
    data.csv_data=df_spl 
    data.csv_summary=df_sum
    print('SPL indices calculated')
    
    
def calculate_acoustic_print():    
    X,y,nmds=analyzis.ac_print(df.md)  
    data.csv_summary['nmds']=nmds.tolist()
    data.csv_summary['ac_print']=X.tolist()
    print('Acoustic print calculated succesfully')

###########################################
### Save Data #############################
###########################################

def save_wav():
    p.save=askdirectory(title="Select folder to save file")   
    print('Write the name of the file in the next line and press enter') 
    w.txt=input()                                         
    filename= p.save + '/' + w.txt +'.wav'
    sound.write(filename, data.wavfs, data.wav, bit_depth=16)
    print('File created:')
    print(filename)
    
def save_csv():
    p.save=askdirectory(title="Select folder to save file")    
    if df.md.empty:
        print('No metadata loaded') 
    else:
        filename= p.save + '/' + 'meta_data'
        df.md.to_csv(filename, Esep='\t', header=True, index=False, encoding='utf-8')
        print('File created:')
        print(filename)   
        
    if data.csv_data.empty:
        print('No raw data loaded') 
    else:
        filename= p.save + '/' + 'raw_data'
        data.md.to_csv(filename, Esep='\t', header=True, index=False, encoding='utf-8')
        print('File created:')
        print(filename)   
        
    if data.csv_summary.empty:
        print('No summary of set of recorders loaded') 
    else:
        filename= p.save + '/' + 'summary' 
        data.csv_summary.to_csv(filename, Esep='\t', header=True, index=False, encoding='utf-8')
        #data.csv_summary.to_excel('summary.xlsx')
        print('File created:')
        print(filename)         

### Checkbox ###########################################################
def activ_spec_vars():
    ch=ch_rec.get()
    if ch==1:
        db.configure(state='normal');fmine.configure(state='normal') 
        fmaxe.configure(state='normal'); tlene.configure(state='normal') 
        bstd.configure(state='normal') ; bp.configure(state='normal') 
        db.delete(0,'end'); db.insert(0, "0"); 
        fmine.delete(0,'end'); fmine.insert(0, "100"); 
        fmaxe.delete(0,'end'); fmaxe.insert(0, "10000")
        tlene.delete(0,'end'); tlene.insert(0, "5");  
        bstd.delete(0,'end'); bstd.insert(0, "0.8");  
        bp.delete(0,'end'); bp.insert(0, "0.1")
        # Fmax
        
    elif ch==0:     
        db.configure(state='disabled'); fmine.configure(state='disabled')
        fmaxe.configure(state='disabled');tlene.configure(state='disabled')                     
        bstd.configure(state='disabled') ; bp.configure(state='disabled')
        
        
##########################################################################
##########################################################################
#########                          GUI                     ###############
##########################################################################
##########################################################################
        
root=tk.Tk()
cwd = os.getcwd()
root.geometry('500x550')    

##### Frame buttons###############################################
frame_bf=ttk.Frame(root); frame_bf.grid(row=1,column=1)
lbf=ttk.Label(frame_bf,text="Load audio files"); lbf.grid(row=0,column=1)

#Folder of recorders
b_read_recs=tk.Button(frame_bf,text="Load set of recorders",padx=10,pady=5,fg="white",bg="#263D42", command=prep_recs)
b_read_recs.grid(row=3,column=1)


#Recorder
b_read_folder=tk.Button(frame_bf,text="Load one recorder",padx=10,pady=5,fg="white",bg="#263D42", command=get_data_files)
b_read_folder.grid(row=2,column=1)

#File
b_read_fi=tk.Button(frame_bf,text="Load file",padx=10,pady=5,fg="white",bg="#263D42", command=read_file)
b_read_fi.grid(row=1,column=1)

b_days=tk.Button(frame_bf,text="Select days",padx=10,pady=5,fg="white",bg="#263D42", command=sel_days)
b_days.grid(row=4,column=1)

#Functions

frame_bfi=ttk.Frame(frame_bf)
frame_bfi.grid(row=6,column=1,pady=50)
lbf=ttk.Label(frame_bfi,text="Functions for datasets"); lbf.grid(row=0,column=1)

b_analyze_day=tk.Button(frame_bfi,text="Spectrogram (file or one-day)",padx=10,pady=5,fg="white",bg="#263D42", command=one_day_spec)
b_analyze_day.grid(row=3,column=1)

b_read_fi=tk.Button(frame_bfi,text="Regions of interest",padx=10,pady=5,fg="white",bg="#263D42", command=rois_gui)
b_read_fi.grid(row=4,column=1)

b_ind=analyze_recs=tk.Button(frame_bfi,text="Calculate acoustic indices",padx=10,pady=5,fg="white",bg="#263D42", command=calculate_ind)
b_ind.grid(row=5,column=1)


b_spl=analyze_recs=tk.Button(frame_bfi,text="Calculate SPL",padx=10,pady=5,fg="white",bg="#263D42", command=calculate_spl)
b_spl.grid(row=6,column=1)

b_spl=analyze_recs=tk.Button(frame_bfi,text="Acoustic Print",padx=10,pady=5,fg="white",bg="#263D42", command=calculate_acoustic_print)
b_spl.grid(row=7,column=1)
##################################################################

##### Frame buttons save ##########################################
frame_bsave=ttk.Frame(root); frame_bsave.grid(row=2,column=1)
lbf=ttk.Label(frame_bsave,text="Save"); lbf.grid(row=0,column=1)
b_save_wav=tk.Button(frame_bsave,text="Save audio file (.wav)",padx=10,pady=5,fg="white",bg="#263D42", command=save_wav)
b_save_wav.grid(row=1,column=1)
b_save_csv=tk.Button(frame_bsave,text="Save data (.cvs)",padx=10,pady=5,fg="white",bg="#263D42", command=save_csv)
b_save_csv.grid(row=2,column=1)
###################################################################

##### Frame Variables  ###########################################
frame_var=ttk.Frame(root); frame_var.grid(row=1,column=2)
##### Recorder
l0 = tk.Label(frame_var, text="Recorder format"); l0.grid(row=1,column=1)
cbox = ttk.Combobox(frame_var,values=['Audiomoth: aammdd_hhmmss','SongMeter: name_aammdd_hhmmss','Snap: name_aammddThhmmss'],width=29,state='readonly')
cbox.grid(row=1,column=2); cbox.set('Select a recorder format')
## Custom variables check
lc = tk.Label(frame_var, text="Use custom variables"); lc.grid(row=2,column=1)
ch_rec=tk.IntVar();ch1 = tk.Checkbutton(frame_var, command=activ_spec_vars,variable=ch_rec)
ch1.grid(row=2,column=2)
## dB floor
ldb = tk.Label(frame_var, text="dB floor"); ldb.grid(row=3,column=1); 
db = tk.Entry(frame_var,bd=5,state='disabled'); db.grid(row=3,column=2)
l1 = tk.Label(frame_var, text="Fmin"); l1.grid(row=4,column=1)
fmine = tk.Entry(frame_var,bd=5,state='disabled'); fmine.grid(row=4,column=2)
# Fmax
l2 = tk.Label(frame_var, text="Fmax"); l2.grid(row=5,column=1)
fmaxe = tk.Entry(frame_var,bd=5,state='disabled'); fmaxe.grid(row=5,column=2)
## Time seg
l3= tk.Label(frame_var, text="Time segments"); l3.grid(row=6,column=1)
tlene = tk.Entry(frame_var,bd=5,state='disabled'); tlene.grid(row=6,column=2)
## bin_std
l= tk.Label(frame_var, text="bin_std"); l.grid(row=7,column=1)
bstd = tk.Entry(frame_var,bd=5,state='disabled'); bstd.grid(row=7,column=2)
## bin_per
l= tk.Label(frame_var, text="bin_per"); l.grid(row=8,column=1)
bp = tk.Entry(frame_var,bd=5,state='disabled'); bp.grid(row=8,column=2); db.insert(0, "0.25")

#################################################################

root.mainloop()