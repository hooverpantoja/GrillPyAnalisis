# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:56:10 2023

@author: Esteban

Read installation guide and readme file before execution 
"""
import os
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import glob
import numpy as np
from maad import sound #util
import pandas as pd

import gui
from gui import home_path
import analysis
import utils
import obj

# Init
df=obj.metadata(pd.DataFrame(),'_','_','_')
w=obj.widget('ch_rec','flims','txt','tlen','db','bstd','bper')
p=obj.path('_','_')
data=obj.data('wav','wavfs',pd.DataFrame(),pd.DataFrame(),[],pd.DataFrame())
flag=obj.flag('enter','load')
S=obj.spec('_','_','_','_')

analyser = analysis.Analyser()
loader = utils.Loader()
sampler = utils.Sampler()
editor = utils.DataFrameEditor()

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
    """Open GUI to select start/end days and update df.md accordingly."""
    df_new = editor.select_days_gui(df.md, root_window)
    if df_new is None or df_new.empty:
        print('Error')
    else:
        print('Días seleccionados y DataFrame actualizado')
        df.md=[]; df.md=df_new

def resample(): 
    sampler.resample_dataset(df.md)   
    print('Resampleo completado. Archivos guardados en la carpeta de destino.')
    
def show_metadata_df():
    """Show the current metadata DataFrame (df.md) in a new window."""
    try:
        utils.DataFrameEditor.show_dataframe(df.md, title="Metadatos (df.md)", root_window=root_window)
    except Exception as e:
        print(f"Error al mostrar DataFrame: {e}")

def assign_groups_gui():
    """Abrir editor de grupos usando DataFrameEditor (GUI)."""
    try:
        editor.assign_groups_gui(df.md, root_window)
    except Exception as e:
        print(f'Error al abrir el editor de grupos: {e}')
        
def fix_hours_by_intervals():
    """Abrir GUI para corregir horas por intervalos y actualizar df.md."""
    try:
        df_new = editor.fix_hours_by_intervals_gui(df.md, root_window)
        df.md = df_new
        print('Horas corregidas por intervalos y DataFrame actualizado')
    except Exception as e:
        print(f'Error al corregir horas: {e}')
        
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
            _rois, _im_rois = analyser.rois_spec(data,w.flims,ch_rec,w.db,w.bstd,w.bper) 
            
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
        analyser.shortwave(data.wav,data.wavfs,w.flims,w.db) 

    elif flag.load=='folder':
        w.flims,w.samp_len,w.db, _ , _,ch =obj.get_info_widgets(ch_rec,fmine,fmaxe,tlene,df.fs,db,'_','_')
        data.wav,data.wavfs,S.Sxx,S.tn,S.fn=analyser.longwave(df.md,p.load,w.samp_len,w.flims,w.db,ch)
        print('Plot finished')

    elif flag.load=='set':
        print('Spectrogram can not be ploted for set of recorders')
    
### For one recorder or set of recorders #############          
def calculate_ind():         
    print('Calculating indices')       
    X = analyser.ind_per_day(df.md)
    print(str(X))
        
def calculate_spl():     
    print('Calculando SPL')
    df_spl, df_sum = analyser.spl_batch(df.md)
    analyser.plot_spl(df_spl)
    df.md=[]; df.md=df_spl
    data.csv_data=df_spl 
    data.csv_summary=df_sum
    print('Índices SPL calculados')
    
    
def calculate_acoustic_print(): 
    print('Calculando huella acústica promedio')   
    _XX, _df_meta = analyser.ac_print(df.md,save_dir=p.save)
    data.npy_matrixAcousticPrint = _XX
    data.data_analysis = pd.DataFrame(_df_meta)
    save_csv()


def calculate_acoustic_print_by_days():
    print('Calculando huella acústica por días')
    _XX, _df_meta = analyser.ac_print(df.md, by_days=True, save_dir=p.save)
    data.npy_matrixAcousticPrint = _XX
    data.data_analysis = pd.DataFrame(_df_meta)
    save_csv()
    

def calculate_nmds():     
    print('Calculando NMDS de la huella acústica')
    _X, _y, _pts, _df_meta = analyser.calculate_nmds()
    data.data_analysis = pd.DataFrame(_df_meta)

###########################################
### Guardar datos #########################
###########################################

def get_workspace_dir():
    """Return the workspace directory from the GUI field or Desktop if empty."""
    try:
        val = save_dir.get().strip()
        return val if val else home_path
    except Exception:
        return home_path

def set_workspace_dir():
    """Open a folder dialog and set the workspace entry and global save path."""
    selected = askdirectory(initialdir=home_path, title="Seleccione carpeta de trabajo")
    if selected:
        try:
            save_dir.delete(0, 'end')
            save_dir.insert(0, selected)
        except Exception:
            pass
        p.save = selected

def save_wav():
    p.save = get_workspace_dir()
    print('Escriba el nombre del archivo en la siguiente línea y presione Enter') 
    w.txt=input()                                         
    filename= p.save + '/' + w.txt +'.wav'
    sound.write(filename, data.wavfs, data.wav, bit_depth=16)
    print('Archivo creado:')
    print(filename)
    
def save_csv():
    p.save = get_workspace_dir()
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
        print('Archivo creado:')
        print(filename)

    if data.npy_matrixAcousticPrint is None:
        print('No hay matriz de huella acústica cargada')
    else:
        filename = p.save + '/' + 'matrixOfAcousticPrints' + '_' + txt_name + '.npy'
        np.save(filename, data.npy_matrixAcousticPrint)
        print('Archivo creado:')
        print(filename)
    
    if data.data_analysis is None:
        print('No hay matriz de huella acústica cargada')
    else:
        filename = p.save + '/' + 'data_analysis' + '_' + txt_name + '.csv'
        data.data_analysis.to_csv(filename, sep='\t', header=True, index=False, encoding='utf-8')
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
frame_btns_load=root_window.insert_subframe(frame_buttons,1,1,"Carga de archivos",pady=5)
root_window.insert_button(frame_btns_load,1,1,"Cargar archivo",read_file)
root_window.insert_button(frame_btns_load,2,1,"Cargar una grabadora",get_data_files)
root_window.insert_button(frame_btns_load,3,1,"Cargar conjunto de grabadoras",prep_recs)

frame_btns_preprocess=root_window.insert_subframe(frame_buttons,2,1,"Pre-procesamiento",pady=5)   
root_window.insert_button(frame_btns_preprocess,1,1,"Resamplear",resample)
root_window.insert_button(frame_btns_preprocess,2,1,"Seleccionar días",sel_days)
root_window.insert_button(frame_btns_preprocess,3,1,"Ver metadatos",show_metadata_df)
root_window.insert_button(frame_btns_preprocess,4,1,"Asignar grupos",assign_groups_gui)
root_window.insert_button(frame_btns_preprocess,5,1,"Corregir hora por intervalos",fix_hours_by_intervals)

frame_btns_process=root_window.insert_subframe(frame_buttons,3,1,"Procesamiento",pady=5)
root_window.insert_button(frame_btns_process,1,1,"Espectrograma (archivo o un día)",one_day_spec)
root_window.insert_button(frame_btns_process,2,1,"Espectrograma (conjunto de grabadoras)",rois_gui)
root_window.insert_button(frame_btns_process,3,1,"Calcular índices por día",calculate_ind)
root_window.insert_button(frame_btns_process,4,1,"Calcular SPL",calculate_spl)
root_window.insert_button(frame_btns_process,5,1,"Huella acústica por sitio",calculate_acoustic_print)
root_window.insert_button(frame_btns_process,6,1,"Huella acústica por días",calculate_acoustic_print_by_days)
root_window.insert_button(frame_btns_process,7,1,"Calcular NMDS de la huella acústica",calculate_nmds)

##### Frame Variables  ###########################################
frame_settings=root_window.insert_frame(1,2)
frame_recorder=root_window.insert_subframe(frame_settings,1,1,"Seleccion de grabadora",pady=10)
cbox=root_window.insert_combobox(frame_recorder,1,1,['Audiomoth: aammdd_hhmmss','Grillo: nombre_aammdd_hhmmss'],
                    width=40,state='readonly', default='Seleccione un formato de grabadora')
frame_var=root_window.insert_subframe(frame_settings,2,1,pady=10)
ch1, ch_rec=root_window.insert_checkbutton(frame_var,1,1,"Variables personalizadas",command=activ_spec_vars)
db=root_window.insert_entry(frame_var,3,1,state='disabled', text="Nivel dB mínimo")
fmine=root_window.insert_entry(frame_var,4,1,state='disabled', text="Fmin (Hz)")
fmaxe=root_window.insert_entry(frame_var,5,1,state='disabled', text="Fmax (Hz)")
tlene=root_window.insert_entry(frame_var,6,1,state='disabled', text="Segmentos de tiempo")
bstd=root_window.insert_entry(frame_var,7,1,state='disabled', text="bin_std")
bp=root_window.insert_entry(frame_var,8,1,state='disabled', text="bin_per")

frame_bsave=root_window.insert_subframe(frame_settings,3,1,pady=5)
root_window.insert_button(frame_bsave,10,1,"Guardar archivo de audio (.wav)",save_wav)
root_window.insert_button(frame_bsave,11,1,"Guardar datos (.csv)",save_csv)

frame_savedir=root_window.insert_subframe(frame_settings,4,1,pady=10)
save_dir=root_window.insert_entry(frame_savedir,12,1,w=30,state='normal',text="Seleccionar carpeta guardado",command=set_workspace_dir)
# Set default to Desktop
save_dir.delete(0,'end')
save_dir.insert(0, home_path)

if __name__ == "__main__":
    root_window.window.mainloop()