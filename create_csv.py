import os
import csv

dir = 'C:/Users/default.DESKTOP-43PHGMT/Desktop/projects/Number Recognition/trainingSet'

with open('C:/Users/default.DESKTOP-43PHGMT/Desktop/projects/Number Recognition/label.csv','w',newline='') as out_file:
    writer = csv.writer(out_file)
    for i in os.listdir(dir):
        for j in os.listdir(dir+'/'+i):
            name = j.split('.')[0]
            label = int(i)
            writer.writerow((name,label))

        