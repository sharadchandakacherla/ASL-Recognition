import os
import time

import cv2
from skimage.transform import resize
import numpy as np
import keras
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten
from keras import models, layers, optimizers
from keras.models import Model
import tensorflow as tf

def preProcessInputImage(imgs, image_dimensions=(50, 50, 3)):
    #debt, train model to accept one maybe image or this might work too, as we could stream a buffer full of images
    X = []
    for img in imgs:
        if img is not None:
            img_file = img.astype(float);
            img_file = resize(img_file, (image_dimensions[0], image_dimensions[1], image_dimensions[2]));
            img_arr = np.asarray(img_file);
            X.append(img_arr)
    return np.asarray(X);

def getModelWithTrainedWeights(pre_trained_model, numclasses, trained_weights_dir, optimizer):
    base_model = pre_trained_model;
    x = base_model.output;
    x = Flatten()(x);
    predictions = Dense(numclasses, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions);

    for layer in base_model.layers:
        layer.trainable = False
    print("compiling untrained model");
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    #loading old weights
    print("loading weights");
    model.load_weights(trained_weights_dir);
    return model;

def getSavedModel(path):
    reconstructed_model = keras.models.load_model(path)
    return reconstructed_model;

def evaluateModelWithLoadedWeights(model, test_images, test_labels):
    loss, acc = model.evaluate(test_images, test_labels, verbose=2);
    print("Untrained model, accuracy: {:5.2f}%".format(100 * acc));

def predictUsingTrainedModel(model,X_test):
    y_pred = model.predict(X_test, verbose=0);
    return y_pred;

def modelFactory(model_name, pretrained_weight_path, input_shape):
    if model_name == "VGG16":
        return VGG16(weights=pretrained_weight_path, include_top=False, input_shape=input_shape);

def runInferenceVGG16(img):
    input_shape = (50, 50, 3);
    img_pre_processed = preProcessInputImage(img,input_shape);
    #cv2.imshow("sad", img_pre_processed.astype(np.uint8));
    #path = "/Users/sharadc/Documents/uic/extra/repos/archive/inference/l1/";
    #path = "/Users/sharadc/Documents/uic/extra/repos/archive/inference/writeL/";
    pre_trained_prefix = "/Users/sharadc/Documents/uic/extra/repos/archive/pre_trained_vgg16"
    weight_path1 = '{0}/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'.format(pre_trained_prefix);
    pretrained_VGG16_model = modelFactory( "VGG16",weight_path1, input_shape);

    optimizer = keras.optimizers.Adam();
    checkpoint_path = "/Users/sharadc/Documents/uic/extra/repos/archive/checkpoints_for_vgg16/vgg16.ckpt";
    loaded_weight_model = getModelWithTrainedWeights(pretrained_VGG16_model, numclasses=30,
                                                     trained_weights_dir=checkpoint_path, optimizer=optimizer);
    y_pred = predictUsingTrainedModel(loaded_weight_model, img_pre_processed);

    Y_pred_classes = np.argmax(y_pred, axis=1)
    map_characters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
                      12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
                      23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space', 29: 'other'}
    y = [map_characters[x] for x in Y_pred_classes]
    print(y);
    return y;


# path ="/Users/sharadc/Documents/uic/extra/repos/archive/inference/kNew/knew.png";
# path ="/Users/sharadc/Documents/uic/extra/repos/archive/inference/c1/C/opencv_frame_1.png";
# img_file = cv2.imread(path);
# img_file=img_file.astype(float);
# # X = np.asarray(X)
# ls = [img_file];
# runInferenceVGG16(np.array(ls));
# cv2.waitKey(1)