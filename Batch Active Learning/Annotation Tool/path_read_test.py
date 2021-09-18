import json
import os
import glob


image_paths = glob.glob('static/Test_Set/*.png')
print(len(image_paths))
fp = open('static/1_180.json', 'r')
m = json.load(fp)


def globRead_idx_file_name_maps(image_paths):
    ordered_file_name_list = []
    # key: image_paths[i].split('/')[-1].split('__')[0] ==> index_read
   

    for i in range(len(image_paths)):
        for j in range(len(image_paths)):
            k = image_paths[j].split('/')[-1].split('__')[0]
            if int(k) == int(i):
                ordered_file_name_list.append(image_paths[j].split('/')[-1])
                break

    return ordered_file_name_list


ordered_file_name_list = globRead_idx_file_name_maps(image_paths)
print(ordered_file_name_list)