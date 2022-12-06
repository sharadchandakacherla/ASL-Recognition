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

print(tf.__version__)
#loading images
path ="/Users/sharadc/Documents/uic/extra/repos/archive/inference/l1/";
imageSize=50
X=[]
y=[]
import time;

for folderName in os.listdir(path):
    if folderName==".DS_Store":
        continue
    label=folderName
    for image_filename in os.listdir(path + folderName):
        img_file = cv2.imread(path + folderName + '/' + image_filename)
        if img_file is not None:
            img_file=img_file.astype(float);
            img_file = resize(img_file, (imageSize, imageSize, 3))
            time1 = str(time.time());
            cv2.imwrite("image_{0}_{1}.jpeg".format(time1,label),img_file.astype(np.uint8))
            img_arr = np.asarray(img_file)
            X.append(img_arr)
            y.append(label)
X = np.asarray(X)
y = np.asarray(y)

#loading model
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

# pretrained_ResNet50V2_model= ResNet50V2(weights = 'imagenet', include_top=False, input_shape=(imageSize, imageSize, 3));
# optimizer = keras.optimizers.Adam()
# checkpoint_path = "/Users/sharadc/Documents/uic/extra/repos/archive/checkpoints_resnetV2/cp_resnetv2.ckpt";

pre_trained_prefix = "/Users/sharadc/Documents/uic/extra/repos/archive/pre_trained_vgg16"
weight_path1 = '{0}/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'.format(pre_trained_prefix);
pretrained_VGG16_model= VGG16(weights = weight_path1, include_top=False, input_shape=(imageSize, imageSize, 3));
optimizer = keras.optimizers.Adam()
checkpoint_path = "/Users/sharadc/Documents/uic/extra/repos/archive/checkpoints_for_vgg16/vgg16.ckpt";

loaded_weight_model = getModelWithTrainedWeights(pretrained_VGG16_model, numclasses=30, trained_weights_dir = checkpoint_path, optimizer= optimizer);
y_pred = predictUsingTrainedModel(loaded_weight_model,X)
saved_model_path ="/Users/sharadc/Documents/uic/extra/repos/archive/saved_models/asl_alpabet_cnn.h5";
# loaded_weight_model = getSavedModel(saved_model_path);
# y_pred = predictUsingTrainedModel(loaded_weight_model,X)
print(y_pred)
# ytest = np.array([19,14,3,11])
# from keras.utils.np_utils import to_categorical
# y_testHot = to_categorical(ytest, num_classes = 30)
#
# Y_true = np.argmax(ytest,axis = 1)

Y_pred_classes = np.argmax(y_pred,axis = 1)
map_characters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space', 29: 'other'}
y=[map_characters[x] for x in Y_pred_classes]
print(y)
# Y_pred_classes = np.argmax(y_pred,axis = 1)
# Y_true = np.argmax(y_testHot,axis = 1)
# confusion_mtx = confusion_matrix(Y_true, Y_pred_classes);
# print(confusion_mtx);