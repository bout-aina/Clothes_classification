from __future__ import division, print_function
#operation system
import os


#numpy  C'est une bibliothèque Python qui fournit un objet tableau multidimensionnel
#package fondamental pour le calcul scientifique
import numpy as np

# Keras #Keras is an open-source software bibliotheque qui fournit une interface Python pour les réseaux de neurones artificiels
from keras.models import load_model

import keras.utils as image

# Flask utils
#The Request, in Flask, is an object that contains all the data sent from the Client to Server.
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


# Image loader
#computer version
import cv2

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'boutaina.h5'

# Load your trained model
model = load_model(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')



def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(28, 28))
    img_array = np.asarray(img)
    x = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
    result = int(img_array[0][0][0])
    print(result)
    if result > 128:
      img = cv2.bitwise_not(x)
    else:
      img = x
    img = img/255
    img = (np.expand_dims(img,0))

    preds =  model.predict(img)
    print(preds)
    return preds


@app.route('/', methods=['GET'])
def home():
    # Main page
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        # argmax quelle étiquette a la valeur de confiance la plus élevée
        predicted_label = np.argmax(preds)
        result = class_names[predicted_label]
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

