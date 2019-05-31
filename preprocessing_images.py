import os
import shutil

orig_dir = 'C:/Users/default.DESKTOP-43PHGMT/Desktop/projects/Number Recognition/trainingSet'
processed_dir = 'C:/Users/default.DESKTOP-43PHGMT/Desktop/projects/Number Recognition/processed_images'

for i in os.listdir(orig_dir):
    for j in os.listdir(orig_dir+'/'+i):
        shutil.copy(orig_dir+'/'+i+'/'+j,processed_dir)
