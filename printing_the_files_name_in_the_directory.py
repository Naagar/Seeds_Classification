##  printing the files name in the directory


import os

for root, dirs, files in os.walk("seeds_dataset/"):
    for filename in files:
        print(filename)