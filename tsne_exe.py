import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.pyplot import imshow
from PIL import Image
from sklearn.manifold import TSNE
import tsne_featurep_generator

def CreateTsne():
    images, pca_features, fs = pickle.load(open('features.p', 'rb'))

    #num_images_to_plot = len(images)

    #if len(images) > num_images_to_plot:
    #    sort_order = sorted(random.sample(range(len(images)), num_images_to_plot))
    #    images = [images[i] for i in sort_order]
    #    pca_features = [pca_features[i] for i in sort_order]

    X = np.array(pca_features)
    tsne = TSNE(n_components=2, learning_rate=150, perplexity=10, angle=0.2, verbose=2).fit_transform(X)

    tx, ty = tsne[:,0], tsne[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

    width = 2000
    height = 1500
    max_dim = 100

    #creating canvas
    full_image = Image.new('RGBA', (width, height))
    for img, x, y in zip(images, tx, ty):
        tile = Image.open(img)
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))
    full_image = full_image.convert("RGB")
    pickle.dump(full_image, open('result.jpg', 'wb'))
    #plt.figure(figsize = (16,12))
    #plt.imshow(full_image)
    #plt.title("results")

    #plt.savefig("C:/Users/ogoul/Desktop/result.png")
    #plt.show()

    return full_image
