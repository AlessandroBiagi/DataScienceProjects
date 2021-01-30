import os, shutil
import tensorflow as tf
import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import cv2

class DataOrganizer:
    """
    """
    def __init__(self, base_dir, category_list, type_dir_list):
        """
        """
        self.base_dir = base_dir
        self.category_list = category_list
        self.type_dir_list = type_dir_list
    
    def creating_step_directories(self):
        """
        """
        for step in self.type_dir_list:
            step_dir = os.path.join(self.base_dir, step)
            if not step_dir:
                os.mkdir(step_dir)
            yield step_dir
        
    def creating_dataset_directories(self):
        """
        """
        for type_dir in self.type_dir_list:
            for category in self.category_list:
                new_cat_dir = os.path.join(type_dir, category)
                if not new_cat_dir:
                    os.mkdir(new_cat_dir)
    
    def selecting_data_names(self):
        """
        """
        for category in self.category_list:
            category_dir = self.base_dir + 'images/' + category
            element_names = os.listdir(category_dir)
            element_full_names = [os.path.join(category_dir, x) for x in element_names]
            
            for type_dir in self.type_dir_list:
                if type_dir == 'train':
                    category_train_names = element_full_names[0:700]
                elif type_dir == 'validation':
                    category_validation_names = element_full_names[700:850]
                elif type_dir == 'test':
                    category_test_names = element_full_names[850:1000]
                    
            category_names_list = [category_train_names, category_validation_names, category_test_names]
            yield category_names_list
            
    def creating_datasets(self, names_lists):
        """
        """
        for position_name, names_list in enumerate(names_lists):
            category = names_list[position_name][0].split("/")[-2]
            for position_type, names_type_list in enumerate(names_list):  
                if position_type == 0:
                    type_dir = 'train'
                elif position_type == 1:
                    type_dir = 'validation'
                elif position_type == 2:
                    type_dir = 'test'
                for name in names_type_list:
                    img_name = name.split("/")[-1]
                    dst_dir_type = os.path.join(self.base_dir, type_dir)
                    dst_dir_full = os.path.join(dst_dir_type, category)
                    dst = os.path.join(dst_dir_full, img_name)
                    shutil.copy(name, dst)
                    
class DataPreparator:
    def __init__(self, rescale, type_dir_list, target_size,
             batch_size=20, class_mode='binary'):
        self.rescale = rescale
        self.type_dir_list = type_dir_list
        self.target_size = target_size
        self.batch_size = batch_size
        self.class_mode = class_mode
        
    def image_rescaler(self):
        for type_dir in self.type_dir_list:
            type_dir_gen = ImageDataGenerator(rescale=self.rescale)
            yield type_dir_gen
            
    def type_generator(self, data_generators):
        for position_gen, data_gen in enumerate(data_generators):
            type_generator = data_gen.flow_from_directory(self.type_dir_list[position_gen],
                                                         target_size=self.target_size,
                                                         batch_size=self.batch_size,
                                                         class_mode=self.class_mode)
            yield type_generator

def nn_performance(model):
    acc = model.history['acc']
    val_acc = model.history['val_acc']
    loss = model.history['loss']
    val_loss = model.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
                        directory,
                        target_size=(150, 150),
                        batch_size=batch_size,
                        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i = i + 1
        if i * batch_size >= sample_count:
            break
    return features, labels

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    
    x += 0.5
    x = np.clip(x, 0, 1)
    
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model_filters.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    
    grads = K.gradients(loss, model_filters.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    
    iterate = K.function([model_filters.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    
    img = input_img_data[0]
    # print(deprocess_image(img).shape)
    return deprocess_image(img)

def nn_performance_smoothed(model):
    acc = model.history['acc']
    val_acc = model.history['val_acc']
    loss = model.history['loss']
    val_loss = model.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs,
        smooth_curve(acc), 'bo', label='Smoothed training acc')
    plt.plot(epochs,
        smooth_curve(val_acc), 'b', label='Smoothed validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, smooth_curve(loss), 'bo', label='Smoothed training loss')
    plt.plot(epochs, smooth_curve(val_loss), 'b', label='Smoothed validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


