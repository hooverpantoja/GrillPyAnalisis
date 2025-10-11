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

analyze=analyzis.Analyzer()
loader=load_data.Loader()

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
    df.md=loader.create_df(p.load,flist,rec_format,flag.load)
    data.wav,data.wavfs= sound.load(flist)  
    if df.md.empty:     
        print('Verifique el archivo y el tipo de grabadora')
    else: 
        print('Archivo cargado exitosamente:')
        print(flist)
    
### One recorder - One Folder #######################    
def get_data_files():
    flag.load='folder'
    p.load=askdirectory()
    flist=glob.glob(p.load +'/*.wav') #Reads *.wav files in the selected directory
    flist.sort()
    rec_format=cbox.get()
    df.md=loader.create_df(p.load,flist,rec_format,flag.load)
    if df.md.empty:
        print('Verifique la carpeta y el tipo de grabadora')
    else: 
        print('Carpeta cargada y DataFrame creado')
        _ ,fs = sound.load(flist[0]) 
        df.fs=fs
        loader.plot_folder_stats(df.md)
        
### Set of recorders - Folder containing folders #######################   
def prep_recs():
    flag.load='set' 
    p.load=askdirectory()
    flist=glob.glob(p.load +'/*/*.wav') #Reads *.wav files in the selected directory
    #flist.sort()
    rec_format=cbox.get()
    df.md=loader.create_df(p.load,flist,rec_format,flag.load) 
    if df.md.empty:
        print('Verifique la carpeta y el tipo de grabadora')
    else: 
        data.csv_summary=loader.plot_set_recorders(df.md)
        print('Carpeta cargada y DataFrame creado') 
        print(data.csv_summary)
        
def sel_days():
    df.days, indices = np.unique(df.md.day, return_inverse=True)  
    print('Días a analizar:')
    print(df.days)
    print('Escriba los días que desea analizar en la siguiente línea separados por ":" ej: 20230504:20230507 y presione Enter:')
    w.txt=input()
    df_new=loader.update_df(df.md, w.txt)
    if df_new.empty:        
        print('Error')
    else:
        print('Días seleccionados y DataFrame actualizado')
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
            print('Inicio de cálculo de regiones (ROIs)')
            rois, im_rois=analyzis.rois_spec(data,w.flims,ch_rec,w.db,w.bstd,w.bper) 
            
        else:
            print('Cargue el audio primero')
    elif flag.load=='set' and ch==1:
        print('aqui voy') 
    else:
        print('Cargue un archivo, una carpeta de una grabadora o un conjunto de grabadoras. Este análisis requiere variables personalizadas.')


### For one Recorder ####################################                        
def one_day_spec():
    if flag.load=='file':
        w.flims, _ ,w.db, _ , _ , _ = obj.get_info_widgets(ch_rec,fmine,
                                                           fmaxe,'_',data.wavfs,db,'_','_')
        analyzis.shortwave(data.wav,data.wavfs,w.flims,w.db) 
    
    elif flag.load=='folder':
        w.flims,w.samp_len,w.db, _ , _,ch =obj.get_info_widgets(ch_rec,fmine,fmaxe,tlene,df.fs,db,'_','_')
        
        data.wav,data.wavfs,S.Sxx,S.tn,S.fn=analyzer.longwave(df.md,p.load,w.samp_len,w.flims,w.db,ch)
        print('Plot finished')
        
    elif flag.load=='set':
        print('Spectrogram can not be ploted for set of recorders')
    
### For one recorder or set of recorders #############       
"""def calculate_ind():         
    print('Calculando índices acústicos')      
    df_ind=analyzis.ind_batch(df.md)
    analyzis.plot_acoustic_indices(df_ind)  
    df.md=[]; df.md=df_ind
    data.csv_data=df_ind    
    print('Acoustic indices calculated')"""
    
def calculate_ind():         
    print('Calculating indices')       
    X=analyzer.ind_per_day(df.md)
    print(str(X))
        
def calculate_spl():     
    print('Calculando SPL')
    df_spl, df_sum=analyzer.spl_batch(df.md)    
    analyzer.plot_spl(df_spl) 
    df.md=[]; df.md=df_spl
    data.csv_data=df_spl 
    data.csv_summary=df_sum
    print('Índices SPL calculados')
    
    
def calculate_acoustic_print():    
    X,y,nmds=analyzis.ac_print(df.md)  
    data.csv_summary['nmds']=nmds.tolist()
    data.csv_summary['ac_print']=X.tolist()
    print('Huella acústica calculada exitosamente')

###########################################
### Guardar datos #########################
###########################################

def save_wav():
    p.save=askdirectory(title="Seleccione carpeta para guardar el archivo")   
    print('Escriba el nombre del archivo en la siguiente línea y presione Enter') 
    w.txt=input()                                         
    filename= p.save + '/' + w.txt +'.wav'
    sound.write(filename, data.wavfs, data.wav, bit_depth=16)
    print('Archivo creado:')
    print(filename)
    
def save_csv():
    p.save=askdirectory(title="Seleccione carpeta para guardar el archivo")    
    if df.md.empty:
        print('No hay metadatos cargados') 
    else:
        print('Escriba el nombre del archivo en la siguiente línea y presione Enter') 
        w.txt=input()  
        filename= p.save + '/' + 'meta_data'+ '-'  + w.txt
        df.md.to_csv(filename, sep='\t', header=True, index=False, encoding='utf-8')
        print('Archivo creado:')
        print(filename)   
        
    if data.csv_data.empty:
        print('No hay datos crudos cargados') 
    else:
        print('Escriba el nombre del archivo en la siguiente línea y presione Enter') 
        w.txt=input()  
        filename= p.save + '/' + 'raw_data'
        data.md.to_csv(filename, sep='\t', header=True, index=False, encoding='utf-8')
        print('Archivo creado:')
        print(filename)   
        
    if data.csv_summary.empty:
        print('No hay resumen del conjunto de grabadoras cargado') 
    else:
        print('Escriba el nombre del archivo en la siguiente línea y presione Enter') 
        w.txt=input()  
        filename= p.save + '/' + 'summary' + '-'  + w.txt
        data.csv_summary.to_csv(filename, sep='\t', header=True, index=False, encoding='utf-8')
        #data.csv_summary.to_excel('summary.xlsx')
        print('Archivo creado:')
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
lbf=ttk.Label(frame_bf,text="Cargar archivos de audio"); lbf.grid(row=0,column=1)

#Folder of recorders
b_read_recs=tk.Button(frame_bf,text="Cargar conjunto de grabadoras",padx=10,pady=5,fg="white",bg="#263D42", command=prep_recs)
b_read_recs.grid(row=3,column=1)


#Recorder
b_read_folder=tk.Button(frame_bf,text="Cargar una grabadora",padx=10,pady=5,fg="white",bg="#263D42", command=get_data_files)
b_read_folder.grid(row=2,column=1)

#File
b_read_fi=tk.Button(frame_bf,text="Cargar archivo",padx=10,pady=5,fg="white",bg="#263D42", command=read_file)
b_read_fi.grid(row=1,column=1)

b_days=tk.Button(frame_bf,text="Seleccionar días",padx=10,pady=5,fg="white",bg="#263D42", command=sel_days)
b_days.grid(row=4,column=1)

#Functions

frame_bfi=ttk.Frame(frame_bf)
frame_bfi.grid(row=6,column=1,pady=50)
lbf=ttk.Label(frame_bfi,text="Funciones para conjuntos de datos"); lbf.grid(row=0,column=1)

b_analyze_day=tk.Button(frame_bfi,text="Espectrograma (archivo o un día)",padx=10,pady=5,fg="white",bg="#263D42", command=one_day_spec)
b_analyze_day.grid(row=3,column=1)

b_read_fi=tk.Button(frame_bfi,text="Regiones de interés",padx=10,pady=5,fg="white",bg="#263D42", command=rois_gui)
b_read_fi.grid(row=4,column=1)

b_ind=analyze_recs=tk.Button(frame_bfi,text="Calcular índices acústicos",padx=10,pady=5,fg="white",bg="#263D42", command=calculate_ind)
b_ind.grid(row=5,column=1)


b_spl=analyze_recs=tk.Button(frame_bfi,text="Calcular SPL",padx=10,pady=5,fg="white",bg="#263D42", command=calculate_spl)
b_spl.grid(row=6,column=1)

b_spl=analyze_recs=tk.Button(frame_bfi,text="Huella acústica",padx=10,pady=5,fg="white",bg="#263D42", command=calculate_acoustic_print)
b_spl.grid(row=7,column=1)
##################################################################

##### Frame buttons save ##########################################
frame_bsave=ttk.Frame(root); frame_bsave.grid(row=2,column=1)
lbf=ttk.Label(frame_bsave,text="Guardar"); lbf.grid(row=0,column=1)
b_save_wav=tk.Button(frame_bsave,text="Guardar archivo de audio (.wav)",padx=10,pady=5,fg="white",bg="#263D42", command=save_wav)
b_save_wav.grid(row=1,column=1)
b_save_csv=tk.Button(frame_bsave,text="Guardar datos (.csv)",padx=10,pady=5,fg="white",bg="#263D42", command=save_csv)
b_save_csv.grid(row=2,column=1)
###################################################################

##### Frame Variables  ###########################################
frame_var=ttk.Frame(root); frame_var.grid(row=1,column=2)
##### Recorder
l0 = tk.Label(frame_var, text="Formato de grabadora"); l0.grid(row=1,column=1)
cbox = ttk.Combobox(frame_var,values=['Audiomoth: aammdd_hhmmss','SongMeter: nombre_aammdd_hhmmss','Snap: nombre_aammddThhmmss'],width=29,state='readonly')
cbox.grid(row=1,column=2); cbox.set('Seleccione un formato de grabadora')
## Custom variables check
lc = tk.Label(frame_var, text="Usar variables personalizadas"); lc.grid(row=2,column=1)
ch_rec=tk.IntVar();ch1 = tk.Checkbutton(frame_var, command=activ_spec_vars,variable=ch_rec)
ch1.grid(row=2,column=2)
## Nivel dB mínimo
ldb = tk.Label(frame_var, text="Nivel dB mínimo"); ldb.grid(row=3,column=1); 
db = tk.Entry(frame_var,bd=5,state='disabled'); db.grid(row=3,column=2)
l1 = tk.Label(frame_var, text="Fmin (Hz)"); l1.grid(row=4,column=1)
fmine = tk.Entry(frame_var,bd=5,state='disabled'); fmine.grid(row=4,column=2)
# Fmax
l2 = tk.Label(frame_var, text="Fmax (Hz)"); l2.grid(row=5,column=1)
fmaxe = tk.Entry(frame_var,bd=5,state='disabled'); fmaxe.grid(row=5,column=2)
## Time seg
l3= tk.Label(frame_var, text="Segmentos de tiempo"); l3.grid(row=6,column=1)
tlene = tk.Entry(frame_var,bd=5,state='disabled'); tlene.grid(row=6,column=2)
## bin_std
l= tk.Label(frame_var, text="bin_std"); l.grid(row=7,column=1)
bstd = tk.Entry(frame_var,bd=5,state='disabled'); bstd.grid(row=7,column=2)
## bin_per
l= tk.Label(frame_var, text="bin_per"); l.grid(row=8,column=1)
bp = tk.Entry(frame_var,bd=5,state='disabled'); bp.grid(row=8,column=2); db.insert(0, "0.25")

#################################################################
if __name__ == "__main__":
    root.mainloop()