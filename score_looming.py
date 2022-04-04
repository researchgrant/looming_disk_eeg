# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 10:49:01 2021

@author: gweiss01
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:37:23 2021

@author: gweiss01
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import adi
import seaborn as sns
import os
from datetime import datetime
import cv2
import pdb

vid_folder=r"Z:\Grant\in_vivo_lfp\Looming Disk\100121"#video files
folder_name= r"Z:\Grant\in_vivo_lfp\Looming Disk\100121"#labchart files
file_list = [file for file in os.listdir(folder_name) if file.endswith('.adicht')]

adi_list = [adi.read_file(os.path.join(folder_name,file)) for file in file_list] 
# All id numbering is 1 based, first channel, first block
# When indexing in Python we need to shift by 1 for 0 based indexing
# Functions however respect the 1 based notation ...

# These may vary for your file ...
channel_id = 1
record_id = 1
data = [f.channels[channel_id-1].get_data(record_id) for f in adi_list]
all_movies=[file for file in os.listdir(vid_folder) if file[-4:]==".mp4"]
start_times= [file.channels[0].records[0].record_time.rec_datetime for file in adi_list]
disk_times= [[comment.time for comment in file.channels[0].records[0].comments[:5]] for file in adi_list]
movie_list=['']*len(start_times)
for movie in all_movies:
    movie_start=datetime.strptime(movie.split("loomingdisk")[0],"%Y-%m-%d %H_%M_%S.%f")
    deltas = [abs(movie_start-rec) for rec in start_times]
    match_index= deltas.index(min(deltas))
    if (min(deltas).total_seconds()) < 5:
        movie_list[match_index]=movie
    else:
        print("Movie with no matching signal found")
        
all_scores={}
for i,movie in enumerate(movie_list):
    cap = cv2.VideoCapture(vid_folder+"\\"+movie)
    fps=int(cap.get(cv2.CAP_PROP_FPS))
    fps=30
    bout_scores=[]
    for disk in disk_times[i]:
        print()
        start=int(disk*fps)
        end=int((disk+60)*fps)
        print(start,movie)
        j=start
        cap.set(cv2.CAP_PROP_POS_FRAMES,j)
        while j < end:
            ret, frame = cap.read() #advance and read frame
            # frame=cv2.normalize(frame, None, alpha=80, beta=2*255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            j+=1
            score='none'
            if ret == True:         
                # Display the resulting frame
                cv2.imshow('1:Freeze 2:Dart 3:Rattle 4:None 5:Exclude', frame)
                x=cv2.waitKey(25)
                if x & 0xFF == ord('b'):
                    j-=1
                if x & 0xFF == ord('1'):
                    score='freeze'
                    break
                if x & 0xFF == ord('2'):
                    score='dart'
                    break
                if x & 0xFF == ord('3'):
                    score='rattle'
                    break
                if x & 0xFF == ord('4'):
                    score='none'
                    break
                if x & 0xFF == ord('5'):
                    score='exclude'
                    break
                if x & 0xFF == ord('e'):
                    # When everything done, release 
                    # the video capture object
                    cap.release()
                    # Closes all the frames
                    cv2.destroyAllWindows()
                    userexit.haltandcatchfire() #NOQA
             # Break the loop
            else:
                 break
        bout_scores.append(score)
    for i in range(5-len(bout_scores)):
        bout_scores.append('exclude')
    all_scores[movie]=bout_scores


existing_files = [int(path.split("_")[1]) for path in os.listdir(vid_folder) if "scor3es_" in path]
if existing_files:
    score_index=max(existing_files)+1
else:
    score_index=1
    
pd.DataFrame(all_scores).to_csv(os.path.join(vid_folder,"scores_{}_.csv".format(score_index)))


# When everything done, release 
# the video capture object
cap.release()
   
# Closes all the frames
cv2.destroyAllWindows()
   