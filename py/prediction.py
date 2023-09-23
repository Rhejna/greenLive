import pickle
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle
import lightgbm
import sklearn

import os
# # !pip install opencv-python
# import cv2  # You may need to install OpenCV (cv2) if not already installed


def recommndant_crop(config):
    # loading the model from the saved file
    # pkl_filename = "../models/model_recommandation.pkl"
    ROOT_DIR = os.path.abspath(os.curdir)
    print(ROOT_DIR)
    pkl_filename = os.path.join(ROOT_DIR, 'models/model_recommandation.pkl')
    with open(pkl_filename, 'rb') as f_in:
        model = pickle.load(f_in)

    print("2")
    print(config)
    print("2")
    result = [[value for value in config.values()]]
    print(result)

    # if type(config) == dict:
    #     df = pd.DataFrame(config)
    # else:
    #     df = config

    y_pred = model.predict(result)

    return y_pred


# def predict_mpg(config):
#     # loading the model from the saved file
#     pkl_filename = "models/model_recommandation.pkl"
#     with open(pkl_filename, 'rb') as f_in:
#         model = pickle.load(f_in)
#
#     print("2")
#     print(config)
#     print("2")
#     result = [[value for value in config.values()]]
#     print(result)
#
#     # if type(config) == dict:
#     #     df = pd.DataFrame(config)
#     # else:
#     #     df = config
#
#     y_pred = model.predict(result)
#     print(y_pred)

    return y_pred


def predict_disease(config):
    ##loading the model from the saved file
    model = models.load_model("../models/potatoes.h5")

    IMAGE_SIZE = 256
    BATCH_SIZE = 32
    CHANNELS = 3
    EPOCHS = 50

    class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

    ###################################################################################################################
    ###################################################################################################################
    dataset = tf.keras.preprocessing.image.load_img(
        config,
        target_size=(IMAGE_SIZE, IMAGE_SIZE)
    )

    image_array = tf.keras.preprocessing.image.img_to_array(dataset).astype('uint8')

    # Make predictions without verbose output
    predictions = model.predict(np.expand_dims(image_array, axis=0), verbose=0)

    # Extract the predicted class index and confidence (probability)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index] * 100

    print(predicted_class_index)

    if predicted_class_index == 0 or predicted_class_index == 1:
        y_pred = f"The plant is sick.\n Predicted class label: {class_names[predicted_class_index]}, (Confidence: {confidence:.2f}%)"
    elif predicted_class_index == 2:
        y_pred = f"The plant is healthy.\n Predicted class label: {class_names[predicted_class_index]}, (Confidence: {confidence:.2f}%)"

    return y_pred