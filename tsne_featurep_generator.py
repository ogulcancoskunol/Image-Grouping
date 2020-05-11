import os
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.decomposition import PCA
import pickle
import time
from PIL import Image
import streamlit as st
import keras.backend.tensorflow_backend as tb
import tsne_exe as te

tb._SYMBOLIC_SCOPE.value = True

model = keras.applications.VGG16(weights='imagenet', include_top=True)

#to handle overlap
imageLocation = st.empty()

#to add new image
def add_image(img):
    feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
    feat_extractor.summary()
    images, features, fs = pickle.load(open('features.p', 'rb'))
    x = load_image(img)
    feat = feat_extractor.predict(x)[0]
    images.append(img)
    fs.append(feat)
    features = np.array(fs)
    pickle.dump([images, features, fs], open('features.p', 'wb'))
    te.CreateTsne()
    full_image = pickle.load(open('result.jpg', 'rb'))
    imageLocation.image(full_image, use_column_width=True)

def load_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

#creating features in root directory
def GenerateFeatures(images_path):
    feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
    feat_extractor.summary()

    image_extensions = ['.jpg', '.png', '.jpeg']
    max_num_images = 500

    images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if os.path.splitext(f)[1].lower() in image_extensions]
    if max_num_images < len(images):
        images = [images[i] for i in sorted(random.sample(xrange(len(images)), max_num_images))]

    print("keeping %d images to analyze" % len(images))

    my_bar = st.progress(0)
    fs = []
    for i, image_path in enumerate(images):
        my_bar.progress(i/len(images))
        x = load_image(image_path);
        feat = feat_extractor.predict(x)[0]
        fs.append(feat)

    print('finished extracting features for %d images' % len(images))
    my_bar.empty()
    features = np.array(fs)
    #pca = PCA(n_components=100)
    #pca.fit(features)

    #pca_features = pca.transform(features)

    pickle.dump([images, features, fs], open('features.p', 'wb'))

#---------Streamlit--------
try:
    full_image = pickle.load(open('result.jpg', 'rb'))
    imageLocation.image(full_image, use_column_width=True)
except (OSError, IOError) as e:
    full_image = 3

st.title("Image Grouping")
filename = st.text_input('Enter a folder path:', key="filepath")
st.write('You selected `%s`' % filename)
if(st.button("Create", key="crtreslt")):
    GenerateFeatures(filename)
    fullimg = te.CreateTsne()
    imageLocation.image(fullimg, use_column_width=True)

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

new_photo_path = file_selector()
st.write('You selected `%s`' % new_photo_path)
if(st.button("Add image", key="imageaddnew")):
    st.write("Loading...")
    img = image.load_img(new_photo_path, target_size=model.input_shape[1:3])
    st.image(img)
    add_image(new_photo_path)
    st.write("Done.")
