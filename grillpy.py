# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:56:10 2023

@author: Esteban

Read installation guide and readme file before execution 
"""
import os
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import glob
import numpy as np
from maad import sound #util
import pandas as pd

import gui
from funcs import load_data
from funcs import analyzis
import obj

# Init
df=obj.metadata(pd.DataFrame(),'_','_','_')
w=obj.widget('ch_rec','flims','txt','tlen','db','bstd','bper')
p=obj.path('_','_')
data=obj.data('wav','wavfs',pd.DataFrame(),pd.DataFrame(),[])
flag=obj.flag('enter','load')
S=obj.spec('_','_','_','_')

analyze = analyzis.Analyzer()
loader = load_data.Loader()

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
    df.days, _indices = np.unique(df.md.day, return_inverse=True)  
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
            w.flims, _ , w.db, w.bstd , w.bper, _ = obj.get_info_widgets(ch_rec,fmine,
                                                               fmaxe,'_',data.wavfs,db,bstd,bp)
            #analyzis.rois_spec(data,w.flims,ch_rec,w.db)
            print('Inicio de cálculo de regiones (ROIs)')
            _rois, _im_rois = analyze.rois_spec(data,w.flims,ch_rec,w.db,w.bstd,w.bper) 
            
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
        analyze.shortwave(data.wav,data.wavfs,w.flims,w.db) 

    elif flag.load=='folder':
        w.flims,w.samp_len,w.db, _ , _,ch =obj.get_info_widgets(ch_rec,fmine,fmaxe,tlene,df.fs,db,'_','_')
        data.wav,data.wavfs,S.Sxx,S.tn,S.fn=analyze.longwave(df.md,p.load,w.samp_len,w.flims,w.db,ch)
        print('Plot finished')

    elif flag.load=='set':
        print('Spectrogram can not be ploted for set of recorders')
    
### For one recorder or set of recorders #############          
def calculate_ind():         
    print('Calculating indices')       
    X = analyze.ind_per_day(df.md)
    print(str(X))
        
def calculate_spl():     
    print('Calculando SPL')
    df_spl, df_sum = analyze.spl_batch(df.md)
    analyze.plot_spl(df_spl)
    df.md=[]; df.md=df_spl
    data.csv_data=df_spl 
    data.csv_summary=df_sum
    print('Índices SPL calculados')
    
    
def calculate_acoustic_print():    
    X, _y, nmds, matrixAcousticPrint = analyze.ac_print(df.md)
    # Convert to native lists
    nmds_list = nmds.tolist() if hasattr(nmds, 'tolist') else list(nmds)
    X_list = X.tolist() if hasattr(X, 'tolist') else list(X)
    # Store matrix
    data.npy_matrixAcousticPrint = matrixAcousticPrint
    # If csv_summary has the same number of rows, assign per-row lists; else store as a single-row object
    if isinstance(data.csv_summary, pd.DataFrame) and not data.csv_summary.empty and len(data.csv_summary) == len(nmds_list):
        # Assign lists per-row; ensure object dtype
        data.csv_summary = data.csv_summary.copy()
        data.csv_summary['nmds'] = pd.Series(list(nmds_list), dtype='object').values
        data.csv_summary['ac_print'] = pd.Series(list(X_list), dtype='object').values
    else:
        # Store as single-row object columns
        data.csv_summary = pd.DataFrame({'nmds': [nmds_list], 'ac_print': [X_list]})
    print('Huella acústica calculada exitosamente')

def calculate_acoustic_print_by_days():
    print('Calculando huella acústica por días')
    X, _y, nmds = analyze.ac_print_by_days(df.md)
    # Convert to native lists
    nmds_list = nmds.tolist() if hasattr(nmds, 'tolist') else list(nmds)
    X_list = X.tolist() if hasattr(X, 'tolist') else list(X)
    # Assign per-row if lengths match, else store as a single-row object
    if isinstance(data.csv_summary, pd.DataFrame) and not data.csv_summary.empty and len(data.csv_summary) == len(nmds_list):
        data.csv_summary = data.csv_summary.copy()
        data.csv_summary['nmds'] = pd.Series(list(nmds_list), dtype='object').values
        data.csv_summary['ac_print'] = pd.Series(list(X_list), dtype='object').values
    else:
        data.csv_summary = pd.DataFrame({'nmds': [nmds_list], 'ac_print': [X_list]})
    print('Huella acústica por días calculada exitosamente')

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
    p.save = askdirectory(title="Seleccione carpeta para guardar el archivo")
    print('Escriba el nombre del archivo en la siguiente línea y presione Enter') 
    txt_name=input()

    if df.md.empty:
        print('No hay metadatos cargados')
    else:
        filename = p.save + '/' + 'meta_datos' + '_' + txt_name + '.csv'
        df.md.to_csv(filename, sep='\t', header=True, index=False, encoding='utf-8')
        print('File created:')
        print(filename)

    if data.csv_summary.empty:
        print('No hay resumen del conjunto de grabadoras cargado')
    else:
        filename = p.save + '/' + 'resumen_general_conjunto_grabadoras'+ '_' + txt_name + '.csv'
        data.csv_summary.to_csv(filename, sep='\t', header=True, index=False, encoding='utf-8')
        # data.csv_summary.to_excel('summary.xlsx')
        print('Archivo creado:')
        print(filename)

    if data.npy_matrixAcousticPrint is None:
        print('No hay matriz de huella acústica cargada')
    else:
        filename = p.save + '/' + 'matrixOfAcousticPrints' + '_' + txt_name + '.npy'
        np.save(filename, data.npy_matrixAcousticPrint)
        print('Archivo creado:')
        print(filename)

### Checkbox ###########################################################
def activ_spec_vars():
    ch = ch_rec.get()
    if ch == 1:
        db.configure(state='normal'); fmine.configure(state='normal')
        fmaxe.configure(state='normal'); tlene.configure(state='normal')
        bstd.configure(state='normal'); bp.configure(state='normal')
        db.delete(0,'end'); db.insert(0, "0")
        fmine.delete(0,'end'); fmine.insert(0, "100")
        fmaxe.delete(0,'end'); fmaxe.insert(0, "10000")
        tlene.delete(0,'end'); tlene.insert(0, "5")
        bstd.delete(0,'end'); bstd.insert(0, "0.8")
        bp.delete(0,'end'); bp.insert(0, "0.1")
    else:
        db.configure(state='disabled'); fmine.configure(state='disabled')
        fmaxe.configure(state='disabled'); tlene.configure(state='disabled')
        bstd.configure(state='disabled'); bp.configure(state='disabled')
        
        
##########################################################################
##########################################################################
#########                          GUI                     ###############
##########################################################################
##########################################################################  
root_window=gui.Window()

##### Frame buttons###############################################
frame_buttons=root_window.insert_frame(1,1)
frame_btns_load=root_window.insert_subframe(frame_buttons,4,1,"Carga de archivos",pady=10)
root_window.insert_button(frame_btns_load,1,1,"Cargar archivo",read_file)
root_window.insert_button(frame_btns_load,2,1,"Cargar una grabadora",get_data_files)
root_window.insert_button(frame_btns_load,3,1,"Cargar conjunto de grabadoras",prep_recs)
root_window.insert_button(frame_btns_load,4,1,"Seleccionar días",sel_days)

frame_btns_process=root_window.insert_subframe(frame_buttons,6,1,"Funciones procesamiento",pady=10)
root_window.insert_button(frame_btns_process,1,1,"Espectrograma (archivo o un día)",one_day_spec)
root_window.insert_button(frame_btns_process,2,1,"Espectrograma (conjunto de grabadoras)",rois_gui)
root_window.insert_button(frame_btns_process,3,1,"Calcular índices por día",calculate_ind)
root_window.insert_button(frame_btns_process,4,1,"Calcular SPL",calculate_spl)
root_window.insert_button(frame_btns_process,5,1,"Huella acústica",calculate_acoustic_print)
root_window.insert_button(frame_btns_process,6,1,"Huella acústica por días",calculate_acoustic_print_by_days)

frame_bsave=root_window.insert_subframe(frame_buttons,2,1,"Funciones guardado",pady=10)
root_window.insert_button(frame_bsave,1,1,"Guardar archivo de audio (.wav)",save_wav)
root_window.insert_button(frame_bsave,2,1,"Guardar datos (.csv)",save_csv)
###################################################################

##### Frame Variables  ###########################################
frame_settings=root_window.insert_frame(1,2)
frame_recorder=root_window.insert_subframe(frame_settings,1,1,"Seleccion de grabadora",pady=10)
cbox=root_window.insert_combobox(frame_recorder,1,1,['Audiomoth: aammdd_hhmmss','SongMeter: nombre_aammdd_hhmmss'],
                    width=40,state='readonly', default='Seleccione un formato de grabadora')
frame_var=root_window.insert_subframe(frame_settings,2,1,pady=10)
ch1, ch_rec=root_window.insert_checkbutton(frame_var,1,1,"Variables personalizadas",command=activ_spec_vars)
db=root_window.insert_entry(frame_var,3,1,state='disabled', text="Nivel dB mínimo")
fmine=root_window.insert_entry(frame_var,4,1,state='disabled', text="Fmin (Hz)")
fmaxe=root_window.insert_entry(frame_var,5,1,state='disabled', text="Fmax (Hz)")
tlene=root_window.insert_entry(frame_var,6,1,state='disabled', text="Segmentos de tiempo")
bstd=root_window.insert_entry(frame_var,7,1,state='disabled', text="bin_std")
bp=root_window.insert_entry(frame_var,8,1,state='disabled', text="bin_per")
#################################################################


if __name__ == "__main__":
    root_window.window.mainloop()