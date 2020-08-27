# # Images to csv 


from PIL import Image
import numpy as np
import sys
import os
import csv

# default format can be changed as needed
def createFileList(myDir, format='.png'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

# load the original image
myFileList = createFileList('/images/')

for file in myFileList:
    print(file)
    img_file = Image.open(file)
    # img_file.show()

    # get original image parameters...
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode

    # Make image Greyscale
    # img_grey = img_file.convert('L')
    #img_grey.save('result.png')
    #img_grey.show()

    # Save Greyscale values
    value = np.asarray(img_grey.getdata(), dtype=np.int).reshape(( img_file.size[2], img_grey.size[1], img_grey.size[0]))
    value = value.flatten()
    # print(value)
    with open("seed_dataset.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(value)























# from PIL import Image
# import numpy as np
# import sys
# import os
# import csv

# ## Useful function
# def createFileList(myDir, format='.png'):
# 		fileList = []
# 		print(myDir)
# 		for root, dirs, files in os.walk(myDir, topdown=False):
# 			for name in files:
# 				if name.endswith(format):
# 					fullName = os.path.join(root, name)
# 					fileList.append(fullName)
# 	return fileList

# mydir='images'

# ##load the original image

# fileList = createFileList(mydir)

# for file in fileList:
# 	print(file)
# 	img_file = Image.open(file)
# 	# img_file.show()

# 	# get original image parameters...
# 	width, height = img_file.size
# 	format = img_file.format
# 	mode = img_file.mode

# 	# Make image Greyscale
# 	img_grey = img_file.convert('L')
# 	#img_grey.save('result.png')
# 	#img_grey.show()

# 	# Save Greyscale values
# 	value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
# 	value = value.flatten()
# 	print(value)

# with open("img_pixels.csv", 'a') as f:
#     writer = csv.writer(f)
#     writer.writerow(value)























# from PIL import Image
# import numpy as np
# import sys
# import os
# import csv

# def creatFilelist(myDir, format='.png'):
# 	fileList = []
# 	print(myDir)

# 	for root, dirs, files in os.walk(myDir, topdown=False):
# 		for name in files:
# 			if name.endswith(format):
# 				fullNmae = os.path.join(root, name)
# 				fileList.append(fullNmae)
# 	return fileList

# # loading the original images
# myFileList = creatFilelist('/images')

# for file in myFileList:
# 	print(file)
# 	img_file = Image.open(file)
# 	width, height = img_file.size
# 	print(width,height)

# 	format = img_file.format
# 	mode = img_file.mode

# 	value = np.asarray(img_file.getdata(), dtype=np.int),reshape(img_file.shape[2], img_file.shape[1], img_file.shape[0])
# 	value = value.flatten()

# 	# print(value)

# 	with open('images_to_csv.csv', 'a') as f:
# 		writer = csv.writer(f)
# 		writer.writerow(value)

# # img = np.array(Image.open("image.jpg"))


# # import csv

# # def csvWriter(fil_name, nparray):
# #   example = nparray.tolist()
# #   with open(fil_name+'.csv', 'w', newline='') as csvfile:
# #     writer = csv.writer(csvfile, delimiter=',')
# #     writer.writerows(example)

# # csvWriter("myfilename", img)