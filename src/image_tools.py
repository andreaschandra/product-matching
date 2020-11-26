import cv2
import matplotlib.pyplot as plt
import keras
from keras import layers


def check_image(images, path_image = '../../ndsc_data/training_img/training_img/', figsize = (10,8), title = None):
  # input : list of images (exactly 2 title of image)
  img1 = cv2.imread(path_image+images[0])
  img2 = cv2.imread(path_image+images[1])
  
  
  fig, axs = plt.subplots(1,2, figsize=figsize )
  
  axs[0].imshow(img1)
  axs[1].imshow(img2)
  
  if title.any() :
    t1 = []
    for t in title:
      t2 = t.split(' ')
      chnks = [t2[x:x+5] for x in range(0, len(t2), 5)]
      chunks = [' '.join(z) for z in chnks]
      t2 = '\n'.join(chunks)
      t1.append(t2)
      
    axs[0].set_title(t1[0])
    axs[1].set_title(t1[1])
    

def auto_encoderv1(INPUT_DIM=44):
  input_img = keras.Input(shape=(INPUT_DIM, INPUT_DIM, 3))

  x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
  x = layers.MaxPooling2D((2, 2), padding='same')(x)
  x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
  x = layers.MaxPooling2D((2, 2), padding='same')(x)
  x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
  encoded = layers.MaxPooling2D((2, 2), padding='same', name = 'enc')(x)

  x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
  x = layers.UpSampling2D((2, 2))(x)
  x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
  x = layers.UpSampling2D((2, 2))(x)
  x = layers.Conv2D(32, (3, 3), activation='relu')(x)
  x = layers.UpSampling2D((2, 2))(x)
  decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same', name = 'dec')(x)

  auto_enc = keras.Model(input_img, decoded)
  # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
  
  return auto_enc


def auto_encoderv2(INPUT_DIM=76):
  input_img = keras.Input(shape=(INPUT_DIM, INPUT_DIM, 3))

  x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
  x = layers.MaxPooling2D((2, 2), padding='same')(x)
  x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
  x = layers.MaxPooling2D((2, 2), padding='same')(x)
  x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
  x = layers.MaxPooling2D((2, 2), padding='same')(x)
  x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
  encoded = layers.MaxPooling2D((2, 2), padding='same', name = 'enc')(x)

  x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
  x = layers.UpSampling2D((2, 2))(x)
  x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
  x = layers.UpSampling2D((2, 2))(x)
  x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
  x = layers.UpSampling2D((2, 2))(x)
  x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
  x = layers.UpSampling2D((2, 2))(x)
  x = layers.Conv2D(64, (3, 3), activation='relu')(x)
  x = layers.UpSampling2D((2, 2))(x)
  decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same', name = 'dec')(x)

  auto_enc = keras.Model(input_img, decoded)
  # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
  
  return auto_enc


def auto_encoderv2(INPUT_DIM=76):
  input_img = keras.Input(shape=(INPUT_DIM, INPUT_DIM, 3))

  x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
  x = layers.MaxPooling2D((2, 2), padding='same')(x)
  x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
  x = layers.MaxPooling2D((2, 2), padding='same')(x)
  x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
  x = layers.MaxPooling2D((2, 2), padding='same')(x)
  x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
  encoded = layers.MaxPooling2D((2, 2), padding='same', name = 'enc')(x)