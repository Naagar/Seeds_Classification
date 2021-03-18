import streamlit as st
from PIL import Image

import style

st.title('Pytorch style transfer ')

img = st.sidebar.selectbox(

 		'Select Image', 
 		('amber.jpg', 'cat.png')
 		)

style_name = st.sidebar.selectbox(

 		'Select Style', 
 		('candy', 'rain-princess', 'mosaic', 'udnie')
 		)