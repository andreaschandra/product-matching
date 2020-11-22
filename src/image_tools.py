import cv2
import matplotlib.pyplot as plt


def check_image(images, path_image = '../../ndsc_data/training_img/training_img/', figsize = (10,8), title = None):
  # input : list of images (exactly 2 title of image)
  img1 = cv2.imread(path_image+images[0])
  img2 = cv2.imread(path_image+images[1])
  
  
  fig, axs = plt.subplots(1,2, figsize=figsize )
  
  axs[0].imshow(img1)
  axs[1].imshow(img2)
  
  if title :
    axs[0].set_title(title)
    axs[1].set_title(title)