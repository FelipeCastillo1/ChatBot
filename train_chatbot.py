import nltk
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

import json
import pickle

#AQUÍ SE AÑADEN LAS LIBRERIAS PARA CREAR NUESTRA CNN

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD

# definición de las variables a utilizar

lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_words = ['?','!']

#leer o cargar nuestro archivo json (intents2.json) y esto servirá
#para que nuestra red aprenda que debe respoder (chatbot)

data_file = open('comprarTarjetas.json').read()
intents = json.loads(data_file)
#print(intents)

#preprocesar los datos de intents

for intent in intents['intents']:
    for pattern in intent['patterns']:

        # tokenize cada palabra
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # agregue cada palabra al cuerpo
        documents.append((w, intent['tag']))

        # agregue cada lista a nuestra clase
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
  

print(words)
print(classes)

#transforma todas las expresiones de word a minúsculas con lower()
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

print(words)

# se crean los siguientes archivos serializables y se guardan 
#en la misma ruta que el proyecto

pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

# Creación del training data o datos de entrenamiento 
training = []
# creación de  un array vacío que almacenará la salida
output_empty = [0] * len(classes)
for doc in documents:
    # bag of words o ocurrencia de cada palabra
    bag = []
    # lista de tokens
    pattern_words = doc[0]
    # lemmatizazone de token
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # # si la palabra hace un match o coincide, ingreso 1, de lo contrario 0
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

training = np.array(training)
# creación de entrenamiento y conjuntos de prueba: X - patrones, Y - intenciones
train_x = list(training[:,0])
train_y = list(training[:,1])

print("Training data creado")
# creación del modelo
# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 
# 3rd output layer contains number of neurons  
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model2.h5', hist)

print("Nuestro modelo ha sido creado")