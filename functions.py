import os
import time
import cv2
import numpy as np
from tensorflow.python.platform import gfile
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Lambda, Input, Conv2D, Conv2DTranspose, concatenate, MaxPooling2D
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import preprocess_input
from keras.layers import Dense, Input
from keras.models import Model, load_model
from skimage.filters import rank
from skimage.morphology import disk
#from keras.utils import print_summary

# Loading Images
def load_images(path):
	images = []
	names = []
	for filename in os.listdir(path):
		img = cv2.imread(os.path.join(path,filename),cv2.IMREAD_GRAYSCALE)
		if img is not None:
			images.append(img)
			names.append(filename)
	infer_images = np.array(images)
	return infer_images,names

# Modifying Images
def resize_images_of(X):
    X = np.array([cv2.resize(image, dsize=(320, 320), interpolation=cv2.INTER_CUBIC) for image in X])
    X = np.array([np.expand_dims(a=image, axis=-1) for image in X])
    X = X.astype(dtype=np.uint8)
    return X

# Segmentation
def unet(input_size=(256,256,1)):
    inputs = Input(input_size)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])
    
def do_segmentation(images, segmentor,disk_rad=40, kernel_size=(5, 5), num_iter=3, margin=0):
    equ_images = [rank.equalize(image.squeeze(), selem=disk(radius=disk_rad)) for image in images]
    masks = [segmentor(np.expand_dims(equ_image, axis=[0, -1])) for equ_image in equ_images]
    masks = [cv2.dilate(np.squeeze(mask), kernel=np.ones(kernel_size), iterations=num_iter) for mask in masks]
    images = [image[np.min(np.where(masks[idx]==1)[0]) - margin:np.max(np.where(masks[idx]==1)[0]) + margin,
                    np.min(np.where(masks[idx]==1)[1]) - margin:np.max(np.where(masks[idx]==1)[1]) + margin]
              for idx, image in enumerate(images)]
    images = [cv2.resize(image, dsize=(320, 320), interpolation=cv2.INTER_CUBIC) for image in images]
    images = [np.expand_dims(image, axis=-1) for image in images]
    return np.array(images)
	
# chexNet weights
# https://github.com/brucechou1983/CheXNet-Keras
chexnet_weights = 'chexnet/best_weights.h5'

# chexnet class names
chexnet_class_index_to_name = [
    'Atelectasis',  # 0
    'Cardiomegaly',  # 1
    'Effusion',  # 2
    'Infiltration',  # 3
    'Mass',  # 4
    'Nodule',  # 5
    'Pneumonia',  # 6
    'Pneumothorax',  # 7
    'Consolidation',  # 8
    'Edema',  # 9
    'Emphysema',  # 10
    'Fibrosis',  # 11
    'Pleural_Thickening',  # 12
    'Hernia',  # 13
]

# chexnet class indexes
chexnet_class_name_to_index = {
    'Atelectasis': 0,
    'Cardiomegaly': 1,
    'Effusion': 2,
    'Infiltration': 3,
    'Mass': 4,
    'Nodule': 5,
    'Pneumonia': 6,
    'Pneumothorax': 7,
    'Consolidation': 8,
    'Edema': 9,
    'Emphysema': 10,
    'Fibrosis': 11,
    'Pleural_Thickening': 12,
    'Hernia': 13,
}


def chexnet_preprocess_input(value):
    return preprocess_input(value)

def get_chexnet_model():
    input_shape = (224, 224, 3)
    img_input = Input(shape=input_shape)
    base_weights = 'imagenet'

    # create the base pre-trained model
    base_model = DenseNet121(include_top=False,input_tensor=img_input,input_shape=input_shape,weights=base_weights,pooling='avg')

    x = base_model.output
    # add a logistic layer -- let's say we have 14 classes
    predictions = Dense(14,activation='sigmoid',name='predictions')(x)

    # this is the model we will use
    model = Model(inputs=img_input,outputs=predictions,)

    # load chexnet weights
    model.load_weights(chexnet_weights)

    # return model
    return base_model, model

def get_model():
    # get base model, model
    base_model, chexnet_model = get_chexnet_model()
    # print a model summary
    # print_summary(base_model)

    x = base_model.output
    # Dropout layer
    #x = Dropout(0.2)(x)
    # one more layer (relu)
    x = Dense(512, activation='relu')(x)
    # Dropout layer
    #x = Dropout(0.2)(x)
    #x = Dense(256, activation='relu')(x)
    # Dropout layer
    #x = Dropout(0.2)(x)
    # add a logistic layer -- let's say we have 6 classes
    predictions = Dense(1,activation='sigmoid')(x)

    # this is the model we will use
    model = Model(inputs=base_model.input,outputs=predictions,)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all base_model layers
    for layer in base_model.layers:
        layer.trainable = False

    # initiate an Adam optimizer
    opt = Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=None,decay=0.0,amsgrad=False)

    # Let's train the model using Adam
    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

    return base_model, model
