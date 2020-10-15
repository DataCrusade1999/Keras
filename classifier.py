import numpy as np
import os
import random
import glob
import shutil
import itertools
import silence_tensorflow.auto
import tensorflow as tf
from tensorflow import keras
from random import randint
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.layers import Activation,Dense,Flatten,BatchNormalization,Conv2D,MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)


physical_devices=tf.config.experimental.list_physical_devices('GPU')
print("Num of GPU's Available ==>",len(physical_devices))





os.chdir(r'C:\Users\ashut\source\repos\PythonClassifierApplication1\dogs-vs-cats')
if os.path.isdir('train/dog') is False:
    os.makedirs('train/dog')
    os.makedirs('train/cat')
    os.makedirs('valid/dog')
    os.makedirs('valid/cat')
    os.makedirs('test/dog')
    os.makedirs('test/cat')
    for c in random.sample(glob.glob(r'C:\Users\ashut\source\repos\PythonClassifierApplication1\dogs-vs-cats\train\cat*'),500):
        shutil.move(c,'train/cat')
    for c in random.sample(glob.glob(r'C:\Users\ashut\source\repos\PythonClassifierApplication1\dogs-vs-cats\train\dog*'),500):
        shutil.move(c,'train/dog')
    for c in random.sample(glob.glob(r'C:\Users\ashut\source\repos\PythonClassifierApplication1\dogs-vs-cats\train\cat*'),100):
        shutil.move(c,'valid/cat')
    for c in random.sample(glob.glob(r'C:\Users\ashut\source\repos\PythonClassifierApplication1\dogs-vs-cats\train\dog*'),100):
        shutil.move(c,'valid/dog')
    for c in random.sample(glob.glob(r'C:\Users\ashut\source\repos\PythonClassifierApplication1\dogs-vs-cats\train\cat*'),50):
        shutil.move(c,'test/cat')
    for c in random.sample(glob.glob(r'C:\Users\ashut\source\repos\PythonClassifierApplication1\dogs-vs-cats\train\dog*'),50):
        shutil.move(c,'test/dog')

train_path=r'C:\Users\ashut\source\repos\PythonClassifierApplication1\dogs-vs-cats\train'
valid_path=r'C:\Users\ashut\source\repos\PythonClassifierApplication1\dogs-vs-cats\valid'
test_path=r'C:\Users\ashut\source\repos\PythonClassifierApplication1\dogs-vs-cats\test'


train_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path,target_size=(224,224),classes=['cat','dog'],batch_size=10)
valid_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=valid_path,target_size=(224,224),classes=['cat','dog'],batch_size=10)
test_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path,target_size=(224,224),classes=['cat','dog'],batch_size=10,shuffle=False)


assert train_batches.n==1000
assert valid_batches.n==200
assert test_batches.n==100
assert train_batches.num_classes==valid_batches.num_classes==test_batches.num_classes==2

imgs,labels=next(train_batches)

def plot_images(images):
    fig,axes=plt.subplots(1,10,figsize=(20,20))
    axes=axes.flatten()
    for img,ax in zip(images,axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


plot_images(imgs)

print(labels)



model=Sequential([Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same',input_shape=(224,224,3)),MaxPool2D(pool_size=(2,2),strides=2),
                         Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'),MaxPool2D(pool_size=(2,2),strides=2),Flatten(),
                         Dense(units=2,activation='softmax')])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x=train_batches,validation_data=valid_batches,epochs=8,verbose=2)
if os.path.isdir(r'C:\Users\ashut\source\repos\PythonClassifierApplication1\my_model.h5') is False:
    model.save(r'C:\Users\ashut\source\repos\PythonClassifierApplication1\my_model.h5')

test_imgs , test_labels=next(test_batches)
plot_images(test_imgs)
print(test_labels)

test_batches.classes

predictions=model.predict(x=test_batches,verbose=0)
np.round(predictions)


cm=confusion_matrix(y_true=test_batches.classes,y_pred=np.argmax(predictions,axis=-1))


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


print(test_batches.class_indices)

cm_plot_labels=['cat','dog']

plot_confusion_matrix(cm=cm,classes=cm_plot_labels)

vgg16_model=tf.keras.applications.vgg16.VGG16()

vgg16_model.summary()


model=Sequential()


for layer in vgg16_model.layers[:-1]:
    model.add(layer)

model.summary()

for layer in model.layers:
    layer.trainable=False


model.add(Dense(units=2,activation='softmax'))

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x=train_batches,validation_data=valid_batches,epochs=5,verbose=2)

if os.path.isdir(r'C:\Users\ashut\source\repos\PythonClassifierApplication1\vgg16_cat_dog.h5') is False:
    model.save(r'C:\Users\ashut\source\repos\PythonClassifierApplication1\vgg16_cat_dog.h5')


predictions=model.predict(x=test_batches,verbose=0)

test_batches.classes

cm=confusion_matrix(y_true=test_batches.classes,y_pred=np.argmax(predictions,axis=-1))

test_batches.class_indices

cm_plot_labels=['cat','dog']

plot_confusion_matrix(cm=cm,classes=cm_plot_labels,title='Confusion Matrix')
