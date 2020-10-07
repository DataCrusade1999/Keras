import numpy as np 
import os
import random
import glob
import shutil
import itertools
import silence_tensorflow.auto
from IPython.display import Image
import tensorflow as tf
from tensorflow import keras
from random import randint
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Activation,Dense,Flatten,BatchNormalization,Conv2D,MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)



mobile=tf.keras.applications.mobilenet.MobileNet()


os.chdir(r'C:\Users\ashut\Machine Learning\Sign-Language-Digits-Dataset')
if os.path.isdir('train/0/') is False:
    os.mkdir('train')
    os.mkdir('valid')
    os.mkdir('test')
    for i in range(0,10):
        shutil.move(f'{i}','train')
        os.mkdir(f'valid/{i}')
        os.mkdir(f'test/{i}')

        valid_samples=random.sample(os.listdir(f'train/{i}'),30)
        for j in valid_samples:
            shutil.move(f'train/{i}/{j}',f'valid/{i}')
        test_samples=random.sample(os.listdir(f'train/{i}'),5)
        for k in test_samples:
            shutil.move(f'train/{i}/{k}',f'test/{i}')


train_path=r'C:\Users\ashut\Machine Learning\Sign-Language-Digits-Dataset\train'
valid_path=r'C:\Users\ashut\Machine Learning\Sign-Language-Digits-Dataset\valid'
test_path=r'C:\Users\ashut\Machine Learning\Sign-Language-Digits-Dataset\test'

train_batch=ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=train_path,target_size=(224,224),batch_size=10)

valid_batch=ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=valid_path,target_size=(224,224),batch_size=10)

test_batch=ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=test_path,target_size=(224,224),batch_size=10,shuffle=False)


x=mobile.layers[-6].output
output=Dense(units=10,activation='softmax')(x)

model=Model(inputs=mobile.input,outputs=output)

for layer in model.layers[:-23]:
    layer.trainable=False



model.summary()


model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x=train_batch,validation_data=valid_batch,epochs=10,verbose=2)


def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


  
test_labels=test_batch.classes

predictions=model.predict(x=test_batch,verbose=0)

cm=confusion_matrix(y_true=test_labels,y_pred=predictions.argmax(axis=-1))

print(test_batch.class_indices)

cm_plot_labels=['0','1','2','3','4','5','6','7','8','9']


plot_confusion_matrix(cm=cm,classes=cm_plot_labels,title='Confusion Matrix')


if os.path.isdir(r'C:\Users\ashut\Machine Learning\MobileNet.h5') is False:
    model.save(r'C:\Users\ashut\Machine Learning\MobileNet.h5')