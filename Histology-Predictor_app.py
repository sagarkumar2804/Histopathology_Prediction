import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np

st.set_option('deprecation.showfileUploaderEncoding', False)
	#test_path='/content/drive/MyDrive/dataset/test_img1.jpeg'




# @st.cache(suppress_st_warning=True,allow_output_mutation=True)
# def import_and_predict(image_data, model):
#     image = ImageOps.fit(image_data, (100,100),Image.ANTIALIAS)
#     image = image.convert('RGB')
#     image = np.asarray(image)
#     st.image(image, channels='RGB')
#     image = (image.astype(np.float32) / 255.0)
#     img_reshape = image[np.newaxis,...]
#     prediction = model.predict(img_reshape)
#     return prediction

model = tf.keras.models.load_model('Cancer-Image-Classifier.h5')
class_names=['Breast Cancer', 'Colon adenocarcinoma', 'Colon myelogenous leukemia', 'Lung Squamous cell carcinoma', 'Ovarian Cancer','Lung adenocarcinoma', 'Lung benign tissue']
st.write("""
         ***Histopathology Image Prediction***
         """
         )

st.write("This is a simple image classification web app to predict the class of cancer that Histopathology image belongs to")

file = st.file_uploader("Please upload an image file", type=["jpeg"])

if file is None:
    st.text("You haven't uploaded any image file")
else:
    imageI = Image.open(file)
#     img = keras.preprocessing.image.load_img(
#     imageI, target_size=(180, 180)
#     )
#     img_array = keras.preprocessing.image.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0) # Create a batch
    image = ImageOps.fit(imageI, (180,180),Image.ANTIALIAS)
    image = image.convert('RGB')
    image = np.asarray(image)
    st.image(image, channels='RGB')
    image = (image.astype(np.float32) / 255.0)
    img_reshape = image[np.newaxis,...]

    predictions = model.predict(img_reshape)
    st.write(predictions)
    score = tf.nn.softmax(predictions)
    # prediction = import_and_predict(imageI, model)
    # pred = prediction[0][0]
    st.write( "This image belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))
    #st.write(class_names[np.argmax(score)])
