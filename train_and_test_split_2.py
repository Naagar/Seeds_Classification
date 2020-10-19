import os
import random
from shutil import copyfile
import pandas as pd
import numpy as np
import csv


file_name = 'shuffled_datafile.csv'
csv_file = open('shuffled_datafile.csv','r')
csv_file.readline()

img_source_dir = './images'

train_size = 0.8

# def img_train_test_split(img_source_dir, df, train_size):
    
if not (isinstance(img_source_dir, str)):
    raise AttributeError('img_source_dir must be a string')
        
if not os.path.exists(img_source_dir):
    raise OSError('img_source_dir does not exist')
        
if not (isinstance(train_size, float)):
    raise AttributeError('train_size must be a float')
        
    # Set up empty folder structure if not exists
if not os.path.exists('data1'):
        os.makedirs('data1')
else:
    if not os.path.exists('data1/train'):
        os.makedirs('data1/train')
        # if not os.path.exists('data1/train/broken'):
        #     os.makedirs('data1/train/broken')
        # if not os.path.exists('data1/train/silkcut'):
        #     os.makedirs('data1/train/silkcut')
        # if not os.path.exists('data1/train/pure'):
        #     os.makedirs('data1/train/pure')
        # if not os.path.exists('data1/train/discolored'):
        #     os.makedirs('data1/train/discolored')

    if not os.path.exists('data1/validation'):
        os.makedirs('data1/validation')
        # if not os.path.exists('data1/validation/broken'):
        #     os.makedirs('data1/validation/broken')
        # if not os.path.exists('data1/validation/silkcut'):
        #     os.makedirs('data1/validation/silkcut')
        # if not os.path.exists('data1/validation/pure'):
        #     os.makedirs('data1/validation/pure')
        # if not os.path.exists('data1/validation/discolored'):
        #     os.makedirs('data1/validation/discolored')
            
    
train_subdir = os.path.join('data1/train')
validation_subdir = os.path.join('data1/validation')

        # Create subdirectories in train and validation folders
if not os.path.exists(train_subdir):
    os.makedirs(train_subdir)

if not os.path.exists(validation_subdir):
        os.makedirs(validation_subdir)

train_counter = 0
validation_counter = 0
df_test= pd.DataFrame(columns=['name', 'class'])
df_train = pd.DataFrame(columns=['name', 'class'])
i_test = 0
i_train = 0
        # Randomly assign an image to train or validation folder
for filename, label in csv.reader(csv_file, delimiter=','):
    # print(filename)
    if filename.endswith(".jpg") or filename.endswith(".png"): 
        fileparts = filename.split('.')

        if random.uniform(0, 1) <= train_size:
            if label=='0':
                copyfile(os.path.join(img_source_dir, filename), os.path.join(train_subdir,'discolored', filename ))
                train_counter += 1
                df_train.loc[i_train] = filename , label
                i_train += 1
            if label=='3':
                copyfile(os.path.join(img_source_dir, filename), os.path.join(train_subdir,'silkcut', filename ))
                train_counter += 1
                df_train.loc[i_train] = filename , label
                i_train += 1
            if label=='2':
                copyfile(os.path.join(img_source_dir, filename), os.path.join(train_subdir,'broken', filename ))
                train_counter += 1
                df_train.loc[i_train] = filename , label
                i_train += 1
            if label=='1': 
                copyfile(os.path.join(img_source_dir, filename), os.path.join(train_subdir,'pure', filename ))
                train_counter += 1
                df_train.loc[i_train] = filename , label
                i_train += 1
                 
        else:
            if label=='0':
                copyfile(os.path.join(img_source_dir, filename), os.path.join(validation_subdir,'discolored', filename))
                validation_counter += 1
                df_test.loc[i_test] = filename , label
                i_test += 1
            if label=='3':
                copyfile(os.path.join(img_source_dir, filename), os.path.join(validation_subdir,'silkcut', filename))
                validation_counter += 1
                df_test.loc[i_test] = filename , label
                i_test += 1
            if label=='2':
                copyfile(os.path.join(img_source_dir, filename), os.path.join(validation_subdir,'broken', filename))
                validation_counter += 1
                df_test.loc[i_test] = filename , label
                i_test += 1
            if label=='1':
                copyfile(os.path.join(img_source_dir, filename), os.path.join(validation_subdir,'pure', filename))
                validation_counter += 1
                df_test.loc[i_test] = filename , label
                i_test += 1
df_test.to_csv('test_data_file.csv') 
df_train.to_csv('train_data_file.csv')              
print('Copied ' + str(train_counter) + ' images to data/train/' )
print('Copied ' + str(validation_counter) + ' images to data/validation/')
    

# data_info = pd.read_csv(file_name)
# df = pd.DataFrame.from_csv(file_name)
# data_info = data_info.to_list()
# print(data_info.shape)
# print(data_info.size)
# random.shuffle(data_info)
# print(data_info)
# img_names = pd.read_csv(file_name)
# labels = np.asarray(data_info.iloc[:, 1])
# img_names = np.asarray(data_info.iloc[:, 0])


# test = img_train_test_split(img_source_dir, df, train_size)



# seeds_dataset_labels_file.csv
# titanic_data.head()
# city = pd.DataFrame([['Sacramento', 'California'], ['Miami', 'Florida']], columns=['City', 'State'])
# city.to_csv('city.csv')

## Code to generate DataFrame:
# name_dict = {
#             'Name': ['a','b','c','d'],
#             'Score': [90,80,95,20]
#           }

# df = pd.DataFrame(name_dict)
# list of name, degree, score 
# name = ["aparna", "pankaj", "sudhir", "Geeku"] 
# label = [90, 40, 80, 98] 
  
# dictionary of lists  
# dict = {'name': name,  'label': label}  
     
# df = pd.DataFrame(dict) 
  
# saving the dataframe 
# df.to_csv('train_.csv') 

# random.shuffle(number_list)

# A continuous index value will be maintained 
# across the rows in the new appended data frame. 
# df1.append(df2, ignore_index = True) 

# df = pd.DataFrame(columns=['name', 'class'])
# df = pd.DataFrame(columns=['lib', 'qty1', 'qty2'])
# >>> for i in range(5):
# >>>     df.loc[i] = ['name' + str(i)] + list(randint(10, size=2))

# >>> df
#      lib qty1 qty2
# 0  name0    3    3
# 1  name1    2    4
# 2  name2    2    8
# 3  name3    2    1
# 4  name4    9    6
# copyfile(os.path.join(img_source_dir, filename), os.path.join(validation_subdir, filename + '.' + fileparts[1]))
