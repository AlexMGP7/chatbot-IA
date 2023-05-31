import itertools

# PARTE 1

# Permite el procesamiento de lenguaje natural
import nltk

# nltk.download("punkt")

# Minimizar las palabras
from nltk.stem.lancaster import LancasterStemmer

# Instanciamos el minimizador
stemmer = LancasterStemmer()

# Permite trabajar con arreglos y realizar manipulaciones, conversiones, etc.
import numpy

# Herramienta de deep learning
import tflearn
import tensorflow
from tensorflow.python.framework import ops

# Permite manipular contenido json.
import json

# Permite crear numeros aleatorios
import random

# Permite guardar los modelos de entrenamiento (mejora la velocidad, ya que no hay que entrenar desde 0 varias veces)
import pickle

import requests
import os
import matplotlib as mltp

import dload

print("Primera parte correcta")

# PARTE 2

dload.git_clone("https://github.com/boomcrash/data_bot.git")

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.replace("\\", "//")
with open(dir_path + "\data_bot\data_bot-main/data.json", "r") as file:
    database = json.load(file)

words = []
all_words = []
tags = []
aux = []
auxA = []
auxB = []
training = []
exit = []

try:
    with open("Entrenamiento/brain.pickle", "rb") as pickleBrain:
        all_words, tags, training, exit = pickle.load(pickleBrain)

except:
    for intent in database["intents"]:
        for pattern in intent["patterns"]:
            # Separamos la frase en palabras
            auxWords = nltk.word_tokenize(pattern)
            # Guardamos las palabras
            auxA.append(auxWords)
            auxB.append(auxWords)
            # Guardar los tags
            aux.append(intent["tag"])
    # Simbolos a ignorar
    ignore_words = [
        "?",
        "!",
        ".",
        ",",
        "¿",
        "'",
        '"',
        "$",
        "-",
        ":",
        "_",
        "&",
        "%",
        "/",
        "(",
        ")",
        "=",
        "*",
        "#",
    ]
    for w in auxB:
        if w not in ignore_words:
            words.append(w)
    words = sorted(set(list(itertools.chain.from_iterable(words))))
    # print(words)
    tags = sorted(set(aux))
    print("Here is the list of tags")
    # print(tags)

    # Convertir a minuscula
    all_words = [stemmer.stem(w.lower()) for w in words]
    # print(len(all_words))

    all_words = sorted(list(set(all_words)))

    # Ordenar tags

    tags = sorted(tags)

    # Creamos una salida falsa

    null_exit = [0 for _ in range(len(tags))]
    # print(null_exit)

    for i, document in enumerate(auxA):
        bucket = []
        # minuscula y quitar signos
        auxWords = [stemmer.stem(w.lower()) for w in document if w != "?"]
        "recorremos"
        for w in all_words:
            if w in auxWords:
                bucket.append(1)
            else:
                bucket.append(0)
        exit_row = null_exit[:]
        exit_row[tags.index(aux[i])] = 1
        training.append(bucket)
        exit.append(exit_row)

    # print(training)
    training = numpy.array(training)
    # print(training)
    exit = numpy.array(exit)

    # Crear el archive pickle

    with open("Entrenamiento/brain.pickle", "wb") as pickleBrain:
        pickle.dump((all_words, tags, training, exit), pickleBrain)
