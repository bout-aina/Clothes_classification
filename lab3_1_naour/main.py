# TensorFlow  is an open source framework developed by Google researchers to run machine learning, deep learning
import tensorflow as tf


#numpy  C'est une bibliothèque Python qui fournit un objet tableau multidimensionnel
#package fondamental pour le calcul scientifique
import numpy as np

#pour les figure
import matplotlib.pyplot as plt

###################################
#Step 1 : Download dataset
#Step 2 : Create Model
#Step 3: Train model
#Step 4: Test model
#Step 5 : save model
###################################


#Compiler le modele
# function loss =>  mesure la précision du modèle pendant l'entraînement.
# function optimizer =>  le modèle est mis à jour en fonction des données
# qu'il voit et de sa fonction de perte
#metrics => utilisées pour surveiller les étapes de formation et de test.
##########################################################################


#Step 1 :  Download dataset

#Loads the Fashion-MNIST dataset.
fashion_mnist = tf.keras.datasets.fashion_mnist

#train_images => containing the training data.
#train_labels => for the training data
#test_images => containing the test data
#test_labels=> for the test data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



# voir la forme de image
print(train_images.shape)


# voir image
print(train_images[0])



# Visualiser image avec plot ( une figure)
plt.figure()
plt.imshow(train_images[0])
#des bare coloré
plt.colorbar()
#Configure the grid lines (n'existe pas ).
plt.grid(False)
plt.show()


#Definir size de la figure
plt.figure(figsize=(10,10))
for i in range(25):
    #Add an Axes to the current figure
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    #color mapping pour mapper les données scalaires aux couleurs.
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# Aplatir et Normaliser les données
#valeurs à l'échelle dans une plage de 0 à 1 avant de
# les alimenter au modèle de réseau neuronal.
train_images = train_images / 255.0
test_images = test_images / 255.0

# 2 - Create Modele
model = tf.keras.Sequential([
    #Flatten transforme le format des images d'un tableau bidimensionnel
    # (de 28 par 28 px) en un tableau unidimensionnel 28*28
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    #La première couche Dense compte 128 nœuds
    tf.keras.layers.Dense(128, activation='relu'),
    #La deuxième couche renvoie un tableau logits d'une longueur de 10.
    tf.keras.layers.Dense(10)
])
#Compiler le modele
# function loss =>  mesure la précision du modèle pendant l'entraînement.
# function optimizer =>  le modèle est mis à jour en fonction des données
# qu'il voit et de sa fonction de perte
#metrics => utilisées pour surveiller les étapes de formation et de test.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# 3 - train modele
#flit  "adapte" le modèle aux données d'entraînement
model.fit(train_images, train_labels, epochs=1)


# 4 - test modele
#pour comparez les performances du modèle sur l'ensemble de données de test
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)


#Faire des prédictions
#Attachez une couche softmax pour convertir les sorties
# linéaires du modèle  en probabilités
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

#le modèle a prédit l'étiquette pour chaque image
predictions = probability_model.predict(test_images)
predictions[0]

# quelle étiquette a la valeur de confiance la plus élevée
np.argmax(predictions[0])
test_labels[0]

#Représentez-le graphiquement pour examiner
# l'ensemble complet des 10 prédictions de classe.
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


#Vérifier les prédiction
#voir  la 0ème image, les prédictions et le tableau de prédiction.
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
# plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()



# 5- save models
model.save('models/boutaina.h5')