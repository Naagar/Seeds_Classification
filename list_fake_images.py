import os
import os
import random
from shutil import copyfile
import pandas as pd
import numpy as np
import csv

# specify the img directory path
path = "silkcut"
# pure 1
# discolored 0
# silkcut 3
# broken 2
# list files in img directory
files = os.listdir(path)
print(path)
df_test= pd.DataFrame(columns=['name', 'class'])
img_count = 0
counter = 0
class_name = 3
for file in files:
    # make sure file is an image
    if file.endswith('.jpg'):
    	image_name = file 
    	counter += 1
    	df_test.loc[img_count] = file, class_name
    	img_count += 1
        


df_test.to_csv('data_fake_images_silkcut.csv') 

print('this much ' + str(counter) + ' added to list ' )
print('done')

	
# df_test.loc[img_count] = file, class_name
        # img_count += 1