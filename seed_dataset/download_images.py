import pandas as pd
import urllib.request
import numpy as np

def url_to_jpg(i, url, file_path_for_images, name):

	'''
		args:
			-- i number of image 
			-- url : a URL address of a given image.
			--file_path : where to save the final image.
	'''

	file_name = name
	full_path = '{}{}'.format(file_path_for_images, file_name)
	urllib.request.urlretrieve(url, full_path)

	print('{} saved.'.format(file_name))

	return None


file_name = 'data_file.csv'
file_path_for_images = 'seed_image/'
urls = pd.read_csv(file_name)


for i, name  in enumerate(urls.values):
	url ='https://s3.ap-south-1.amazonaws.com/adtech.monsanto/'
	# print(name)
	url = url + name
	print(url[0])

	# url = url.np.decode('utf-8')
	# print(url)

	url_to_jpg(i, url[0], file_path_for_images, name[0])

