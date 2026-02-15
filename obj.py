# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:29:00 2023

@author: Esteban
"""
class matrixAcousticPrint:
    def __init__(self,matrix):
        self.load=matrix

class flag:
    def __init__(self,enter,load):
        self.enter=enter
        self.load=load

class spec:
    def __init__(self,Sxx,tn,fn,ind):
        self.Sxx=Sxx
        self.tn=tn
        self.fn=fn
        self.ind=ind

class widget:
    def __init__(self,ch_rec,flims,txt,tlen,db,bstd,bper):
        self.ch_rec=ch_rec       
        self.flims = flims
        self.tlen = tlen
        self.db = db
        self.txt = txt
        self.bstd = bstd
        self.bper = bper
        
class metadata:
    def __init__(self, md,fs,path,days):
        self.md = md
        self.fs = fs
        self.path = path
        self.days = days
        
class path:
    def __init__(self, load,save):
        self.load = load
        self.save = save
        
class data:
    def __init__(self, wav, wavfs, csv_summary,csv_data,npy_matrixAcousticPrint,data_analysis):
        self.wav = wav
        self.wavfs = wavfs
        self.csv_summary = csv_summary
        self.csv_data = csv_data
        self.npy_matrixAcousticPrint = npy_matrixAcousticPrint
        self.data_analysis = data_analysis

def get_info_widgets(ch_rec,fmine,fmaxe,tlene,fs,db,bstd,bp):
    if ch_rec == '_': ch= '_'; flims= '_'; samp_len='_'; dbf='_'
    else:
        ch=ch_rec.get()
        if ch==1:
            if tlene == '_': samp_len = '_' 
            else: samp_len=int(tlene.get())
            
            if fmine.get() == '_' and fmaxe.get() == '_': flims='_' 
            else:  flims=[int(fmine.get()),int(fmaxe.get())]
            
            if db == '_': dbf = '_' 
            else: dbf=int(db.get())
            
            if bstd=='_': bstdf = '_'
            else: bstdf=float(bstd.get())
            
            if bp=='_': bpf = '_'
            else: bpf=float(bp.get()) 
            
        elif ch==0: #default values
            samp_len=10 
            flims=[0,fs/2]
            dbf=0
            bstdf=0.5, 
            bpf=0.05,
    return(flims,samp_len,dbf,bstdf,bpf,ch)  

def get_text(textbox):
    if textbox == '_': txt= '_' 
    else:
        txtt=textbox.get("1.0",'end-1c')
        ind=[i for i, ltr in enumerate(txtt) if ltr == '\n']
        init=ind[len(ind)-2]+1; end=len(txtt)-1 #init-> second to last index plus one position to the right; end lenght of the string minus one position to the left 
        txt=txtt[init:end]
        return(txt)