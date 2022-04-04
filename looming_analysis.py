# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:37:23 2021

@author: gweiss01
"""

import numpy as np
import pandas as pd
import scipy, os
from scipy.io import loadmat
from scipy.signal import stft
from matplotlib import pyplot as plt
import adi
import seaborn as sns
import os
from datetime import datetime
import matplotlib
# matplotlib.use('Qt5Agg')


vid_folder=r"Z:\Grant\in_vivo_lfp\Looming Disk\092421"
folder_name= r"Z:\Grant\in_vivo_lfp\Looming Disk\092421"
file_list = [file for file in os.listdir(folder_name) if file.endswith('.adicht')]
animal_list=[file.split(".")[0] for file in file_list]

adi_list = [adi.read_file(os.path.join(folder_name,file)) for file in file_list] 
# All id numbering is 1 based, first channel, first block
# When indexing in Python we need to shift by 1 for 0 based indexing
# Functions however respect the 1 based notation ...

# These may vary for your file ...
channel_id = 1
record_id = 1
data = [f.channels[channel_id-1].get_data(record_id) for f in adi_list]
all_movies=[file for file in os.listdir(vid_folder) if file[-4:]=="looming.mp4"]
start_times= [file.channels[0].records[0].record_time.rec_datetime for file in adi_list]
movie_list=['']*len(start_times)
for movie in all_movies:
    movie_start=datetime.strptime(movie.split("loomingdisk")[0],"%Y-%m-%d %H_%M_%S.%f")
    deltas = [abs(movie_start-rec) for rec in start_times]
    match_index= deltas.index(min(deltas))
    if (min(deltas).total_seconds()) < 5:
        movie_list[match_index]=movie
    else:
        print("Movie with no matching signal found")

disk_times= [[comment.time for comment in file.channels[0].records[0].comments[:5]] for file in adi_list]
fs=adi_list[0].channels[0].fs[0]
window=1

def getSTFT(data,fs):
    time_fft=stft(data,fs=fs,nperseg=fs*window,noverlap=fs*window/2)
    return np.abs(time_fft[2][:,::2])**2

def removeOutliers(fft):
    fft=pd.DataFrame(fft)
    fft[58:63]=np.nan
    fft=fft.fillna(method="ffill",axis=0).fillna(method="bfill",axis=0)
    filtered_fft=fft[fft<(fft.std(axis=1)*2)[:,None]].fillna(method="ffill",axis=1).fillna(method="bfill",axis=1)
    return np.array(filtered_fft)

def powerArea(arr,window):
    return np.array(arr).mean(axis=1)[np.arange(*window,1)].sum()

def avgPwr(arr,window):
    return np.array(arr).mean(axis=1)[np.arange(*window,1)].sum()/(window[1]-window[0])

def quantFear(animal,numeric_scores):
    one_mouse = all_scores[:,animal]
    score=((one_mouse!='none') & (one_mouse!='exclude')).sum()/.05
    return score

def makeTable(norm_ffts):
    table={"Behavior":[],"Frequency":[],"Normalized Power":[],"Animal":[],"Stress":[],"Sex":[],"PSD":[],"Percent Fear Response":[]}
    for i,beh in enumerate(behavior_list):
        for h,animal in enumerate(norm_ffts):
            beh_ffts=animal[all_scores.transpose()[h]==beh]
            for freq,window in freq_ranges.items():
                for fft in beh_ffts:
                    table["Stress"].append(file_list[h].split("_")[1])
                    table["Sex"].append(file_list[h].split("_")[2].split(".")[0])
                    table["Animal"].append(file_list[h].split("_")[0])
                    table["Behavior"].append(beh)
                    table["Frequency"].append(freq)
                    table["Normalized Power"].append(avgPwr(fft,window))
                    table["PSD"].append(fft_array[i,h])
                    table["Percent Fear Response"].append(quantFear(h,numeric_scores))
    table=pd.DataFrame(table)
    return table



fft_list_unf=[getSTFT(d,fs) for d in data]
fft_list=[removeOutliers(fft) for fft in fft_list_unf]

disk_window=(0,60)

baselines=[fft[:,int(disk_times[i][0])-15:int(disk_times[i][0])] for i,fft in enumerate(fft_list)]
baselines=pd.Series(baselines)
df = {i:[ fft[:,int(comment+disk_window[0]):int(comment+disk_window[1])] for comment in disk_times[i]] for i,fft in enumerate(fft_list)}
fft_array=np.array(pd.DataFrame(df))
baseline_means=[fft.mean(axis=1)[:,None] for fft in baselines]
norm_ffts=np.array([np.array(fft/pd.Series(baseline_means)) for fft in fft_array])

freq_ranges={"Low Theta":(2,6),"High Theta":(6,12),"Across Theta":(2,12),"Beta":(15,30),"Low Gamma":(40,70),"High Gamma":(80,120)}
behavior_list=["freeze","dart","rattle","none","exclude"]
colors=sns.color_palette('plasma',len(behavior_list)-1)+[(1,1,1)]
# all_scores=np.array(pd.read_csv(vid_folder+"\scores_grant.csv",index_col=0))

#read in multiple score files to get the mode score
score_files=[file for file in os.listdir(vid_folder) if file.startswith('scores')]
replicate_scores=np.array([np.array(pd.read_csv(vid_folder+"\\"+file,index_col=0)) for file in score_files])
all_scores=scipy.stats.mode(replicate_scores,axis=0)[0][0,:,:]

#give each score an arbitrary numerical value for heatmap
numeric_scores=pd.DataFrame(list(map(np.vectorize(behavior_list.index),all_scores))).transpose()+1
numeric_scores.columns=["Trial "+str(col+1) for col in numeric_scores.columns]
# numeric_scores.index=[str(col+1) for col in numeric_scores.index]
numeric_scores.index=[animal.split("_",1)[1] for animal in animal_list]
numeric_scores=numeric_scores.sort_index()

#create heat map with seaborn
fig,ax=plt.subplots()
sns.heatmap(numeric_scores,cmap=colors,linewidths=3,ax=ax,yticklabels=1,vmin=1,vmax=len(behavior_list)+1, cbar_kws={'drawedges':True})
ax.set_ylabel("Mouse")
ax.xaxis.tick_top() # x axis on top
ax.xaxis.set_label_position('top')
ax.tick_params(length=0)
#relabel the key with the behaviors
cbar = ax.collections[0].colorbar
cbar.set_ticks(np.arange(1.5,len(behavior_list)+1.5,1))
cbar.set_ticklabels(behavior_list)
cbar.solids.set_edgecolor((.4,.4,.4,1))

#remove 'exclude' for the list of behaviors we will analyze
behavior_list=["freeze","dart","rattle","none"]


norm_ffts=norm_ffts.transpose()
table=makeTable(norm_ffts)
# sns.boxplot(x="Frequency",y="Normalized Power",hue="Behavior",data=table,palette='plasma',showfliers=False)

#theta range PSD averaged by animal

table2={"Behavior":[],"Power":[],"Animal":[],"Stress":[],"Sex":[],"Frequency":[]}
for h,animal in enumerate(table["Animal"].unique()):
    for i,beh in enumerate(behavior_list):
        mean=table['PSD'][(table["Behavior"]==beh) & (table["Animal"]==animal)].mean()
        if mean is np.nan:continue
        mean = mean.mean(axis=1)[2:22]
        freqs = range(2,22)
        file=pd.Series(file_list).values[pd.Series(file_list).str.contains(animal)][0].split(".")[0]
        animal,stress,sex=file.split("_")
        for j,power in enumerate(mean):
            table2["Stress"].append(stress)
            table2["Sex"].append(sex)
            table2["Animal"].append(animal)
            table2["Behavior"].append(beh)
            table2["Power"].append(power/mean.mean())
            table2["Frequency"].append(freqs[j])
    for j,mean in enumerate(baseline_means[h][2:22]):
        #add the baseline
        norm_mean=mean/baseline_means[h][2:22].mean()
        table2["Stress"].append(stress)
        table2["Sex"].append(sex)
        table2["Animal"].append(animal)
        table2["Behavior"].append("baseline")
        table2["Power"].append(norm_mean[0])
        table2["Frequency"].append(freqs[j])
        
table2= pd.DataFrame(table2)

colors[-1]=(0,0,0)
fig,ax3=plt.subplots()
table2["Power"]=np.log(table2["Power"])
sns.lineplot(x="Frequency",y="Power",data=table2,ax=ax3,hue="Behavior",palette=colors,hue_order=behavior_list+["Baseline"])
ax3.legend()
# sns.catplot(x="Frequency",y="Normalized Power",hue="Behavior",data=table,palette='plasma',kind="bar",col="Sex",row="Stress")

sns.catplot(data=table,x="Frequency",y="Normalized Power",hue="Percent Fear Response",palette='plasma',kind="bar",col='Sex',row="Stress")
    
plt.show()
