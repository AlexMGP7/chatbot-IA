# PARTE 1

# Permite el procesamiento de lenguaje natural
import nltk

nltk.download("punkt")

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

