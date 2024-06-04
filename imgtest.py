# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 02:13:32 2023

@author: Esteban
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imread, imshow
from skimage.color import rgb2gray
import skimage.morphology as morph
from skimage.measure import label, regionprops, regionprops_table
from maad import sound, util, rois, features
from skimage.transform import downscale_local_mean

dot=1
#square = np.array([[dot,dot,dot],[dot,dot,dot],[dot,dot,dot]])
#square = np.array([[dot,dot],[dot,dot]])
square = np.array([[dot,dot],[dot,dot],[dot,dot]])
e=morph.ellipse(0.5,1)#widh,height
def multi_dil(im, num, element=e):
    for i in range(num):
        im = morph.dilation(im, element)
    return im
def multi_ero(im, num, element=e):
    for i in range(num):
        im = morph.erosion(im, element)
    return im

#%%
#s_name='C:/Users/camil/Escritorio/SENEGAL 2023_04_01/BBG/Extract_BBG_20201202T045835.wav'    
#s_name='C:/Users/camil/Escritorio/SENEGAL 2023_04_01/BBG/Extract_BBG_20201201T235843.wav'  
s_name='C:/Users/camil/Escritorio/PAREX/PAM/Z6-G047/20230415_060000.wav'  
s,fs=sound.load(s_name)
flims=[100, 10000]
db=200

Sxx, tn, fn, ext = sound.spectrogram(s, fs, nperseg=1024, noverlap=512,flims=flims)                  
Sxx_db=20*np.log(Sxx/Sxx.min())
Sxx_db[Sxx_db < db] = 0
#Sxx[Sxx_db < db]=0
#Sxx_db_rmbg, _, _ = sound.remove_background(Sxx_db)
#Sxx_db= sound.smooth(Sxx_db,std=0.05)   
Sxxf=Sxx_db/Sxx_db.max()  
imshow(Sxxf, aspect='auto',origin='lower',extent=[0,tn[-1], 0 ,fn[-1]/1000])
binarized =Sxxf>0.66
imshow(Sxxf, aspect='auto',origin='lower',extent=[0,tn[-1], 0 ,fn[-1]/1000])
imshow(binarized, aspect='auto',origin='lower',extent=[0,tn[-1], 0 ,fn[-1]/1000])
#%%
im=binarized
im=morph.binary_opening(im, footprint=None, out=None)
im=morph.binary_erosion(im, footprint=None, out=None)
im = multi_dil(im, 2)
im = morph.area_closing(im,100)
imshow(im, aspect='auto',origin='lower',extent=[0,tn[-1], 0 ,fn[-1]/1000])

label_im = label(im)
regions = regionprops(label_im)
properties = ['area','coords', 'bbox','convex_area','axis_minor_length']
D=pd.DataFrame(regionprops_table(label_im, Sxxf,properties=properties))

masks = []
ifill=[]
bbox = []
list_of_index = []
roi=[]
roi_mspec=[]
lims=[]
for num, x in enumerate(regions):
    area = x.area
    convex_area = x.convex_area
    min_l=x.axis_minor_length
    if (num!=0 and (area>400) and (min_l > 20)):
        masks.append(regions[num].convex_image)
        ifill.append(regions[num].image_filled)
        bbox.append(regions[num].bbox)   
        list_of_index.append(num)
        tmin=regions[num].bbox[1];tmax=regions[num].bbox[3];
        fmin=regions[num].bbox[0];fmax=regions[num].bbox[2];
        lims.append([tmin,tmax,fmin,fmax])
        #M=np.multiply(regions[num].image_convex,1);
        temp=Sxxf[fmin:fmax,tmin:tmax]*regions[num].image_filled.astype(int)
        roi_mspec.append(temp.mean(1))
        roi.append(temp)
count = len(masks)
print(count)

mask = np.zeros_like(label_im)
for x in list_of_index:
    mask += (label_im==x+1).astype(int)
I  =  Sxxf * mask
imshow(I, aspect='auto',origin='lower')
#%%
n=1
#imshow(roi[n], aspect='auto',origin='lower',extent=[tn[lims[n][0]],tn[lims[n][1]],fn[lims[n][2]],fn[lims[n][3]]])
#plt.plot(fn[lims[n][2]:lims[n][3]],roi_mspec[n])

#%% processing options
for i in range(0, 1, 1):
    im = multi_ero(im, 1)
    im = multi_dil(im, 1)
imshow(im, aspect='auto',origin='lower',extent=[0,tn[-1], 0 ,fn[-1]/1000])
    
im = morph.area_opening(im,50)
imshow(im, aspect='auto',origin='lower',extent=[0,tn[-1], 0 ,fn[-1]/1000])

im=morph.skeletonize(im)
imshow(im, aspect='auto',origin='lower',extent=[0,tn[-1], 0 ,fn[-1]/1000])



seedsx,seedsy=np.where(im==True)
im2=morph.flood_fill(Sxxf, (seedsx[0],seedsy[0]), 1, connectivity=None, tolerance=0.1)
imshow(im2, aspect='auto',origin='lower',extent=[0,tn[-1], 0 ,fn[-1]/1000])

im = multi_dil(im, 1)
imshow(im, aspect='auto',origin='lower',extent=[0,tn[-1], 0 ,fn[-1]/1000])

im = multi_ero(im, 1)
imshow(im, aspect='auto',origin='lower',extent=[0,tn[-1], 0 ,fn[-1]/1000])

im = area_opening(im,500)
imshow(im, aspect='auto',origin='lower',extent=[0,tn[-1], 0 ,fn[-1]/1000])
im=morph.skeletonize(im)

#%%
from sklearn.manifold import MDS
from matplotlib import pyplot as plt
import sklearn.datasets as dt
import seaborn as sns         
import numpy as np
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import sklearn.datasets as dt


def mapData(dist_matrix, X, y, metric, title):
    mds = MDS(metric=metric, dissimilarity='precomputed', random_state=0)
    # Get the embeddings
    pts = mds.fit_transform(dist_matrix)
    # Plot the embedding, colored according to the class of the points
    fig = plt.figure(2, (15,6))
    ax = fig.add_subplot(1,2,1)    
    ax = sns.scatterplot(x=pts[:, 0], y=pts[:, 1],
                         hue=y, palette=['r', 'g', 'b', 'c'])

    # Add the second plot
    ax = fig.add_subplot(1,2,2)
    # Plot the points again
    plt.scatter(pts[:, 0], pts[:, 1])
    
    # Annotate each point by its corresponding face image
    for x, ind in zip(X, range(pts.shape[0])):
        im = x.reshape(64,64)
        imagebox = OffsetImage(im, zoom=0.3, cmap=plt.cm.gray)
        i = pts[ind, 0]
        j = pts[ind, 1]
        ab = AnnotationBbox(imagebox, (i, j), frameon=False)
        ax.add_artist(ab)
    plt.title(title)    
    plt.show()


faces = dt.fetch_olivetti_faces()
X_faces = faces.data
y_faces = faces.target
ind = y_faces < 2
X_faces = X_faces[ind,:]
y_faces = y_faces[ind]
imshow(X_faces, aspect='auto')
dist_euclid = euclidean_distances(X_faces)
mapData(dist_euclid, X_faces, y_faces, True, 'Metric MDS with Euclidean')


