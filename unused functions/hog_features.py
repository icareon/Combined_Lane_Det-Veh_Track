import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog
import random

# Read in our vehicles and non-vehicles
images = glob.glob('../CarND-Vehicle-Detection/test_images/*.jpg')


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:
        # Use skimage.hog() to get features only
        features = hog(img, orientation=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       visualise=False, feature_vector=True)
        return features


# Read in the image
image = mpimg.imread(images[3])
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# Define HOG parameters
orient = 9
pix_per_cell = 8
cell_per_block = 2
# Call our function with vis=True to see an image output
features, hog_image = get_hog_features(gray, orient,
                                       pix_per_cell, cell_per_block,
                                       vis=True, feature_vec=False)

# Plot the examples
fig = plt.figure()
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(hog_image, cmap='gray')
plt.title('HOG Visualization')
fig.savefig('../writeup_imgs/hog_play3.png')
plt.show()


# Read in images of cars and notcars
images = glob.glob('../CarND-Vehicle-Detection/train_images/*/*/*.png')
cars_full = []
notcars_full = []

for image in images:
    if 'non-vehicles' in image:
        notcars_full.append(image)
    else:
        cars_full.append(image)

idcs=random.sample(images,3)

fig=plt.figure(figsize=(7, 7))
for i in range(len(idcs)):
    # Read in the image
    image = mpimg.imread(idcs[i])
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Define HOG parameters
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    features, hog_image = get_hog_features(gray, orient,pix_per_cell, cell_per_block, vis=True, feature_vec=False)

    plt.subplot(3, 2, 2 * (i + 1) - 1)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(3, 2, 2 * (i + 1))
    plt.imshow(hog_image, cmap='gray')
    plt.axis('off')
    plt.tight_layout()

fig.savefig('../writeup_imgs/hog_play4.png')
plt.show()