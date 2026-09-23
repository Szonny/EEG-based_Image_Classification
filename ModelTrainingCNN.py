# -*- coding: utf-8 -*-

import gc
import os
import time
import random
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sys import getsizeof
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, callbacks
from tensorflow.keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder

from datetime import datetime
now = datetime.now().strftime("%m%d_%H%M")

#Definicje i założenia
prefix_file_data = "plikWynikowy"
prefix_file_eti = "plikWynikowyET"
prefix_file_stat = "parametry_"

nazwa_file = "_Dane32Przetworzone"
mark_del_mean = ["AVG", "NoAVG"]
mark_reduction = ["", "ratio", "logratio", "zlogratio", "mean", "zscore" , "RAW"] 
file_stage = ["TRAIN" , "TEST"]

run_no = [[0],[1],[2],[3], [4],[5],[6] , [0,1]] # ostatni element = przbieg testowy
test_run = len(run_no)-1
train_file_no = len(run_no)-1

weight_file = "dogotowywane.weights.h5"

#KONFIGUROWWALNE PARAMETRYY
multifile_data = True

chosen_mark = mark_del_mean[0]+mark_reduction[0]
result_dir="finito"
# 0 trenuj i zapisz niedogotowany model
# 1 trenuj i zapisz dogotowany model
# 2 nie trenuj, tylko przetestuj i zrob wykresy

#Łączenie z dyskiem
#from google.colab import drive
#drive.mount('/content/drive')
#dataSourcePath = '/content/drive/MyDrive/BIAI/'
dataSourcePath = 'data2/'

#plikWynikowyET7_TRAIN_Dane32PrzetworzoneAVGratio

X = []
y_txt = []
input_shape = []
class_amount=0
model: tf.keras.Model | None = None
hist: tf.keras.callbacks.History | None = None
def load_data(curr_run_no=0, if_test_run=False):
    global X, y_txt, X_test, y_test, y, y_train, y_validate, X_validate,  X_train, input_shape, class_amount
    
    X = []
    y_txt = []
    input_shape = []
    class_amount=0
    
    if multifile_data:
      current_stage = file_stage[if_test_run]
      files_postfix = "_"+current_stage+nazwa_file+chosen_mark+".npy"
      parametry_stat = np.load(dataSourcePath+prefix_file_stat+current_stage+chosen_mark+".npy")
      mean = parametry_stat[0]
      odch_std = parametry_stat[1]

      mean = np.transpose(mean, (0,2,3,1))
      odch_std =  np.transpose(odch_std, (0,2,3,1))

      for nr_plikow in run_no[curr_run_no]:
        nx = np.load(dataSourcePath+prefix_file_data +str(nr_plikow)+files_postfix, mmap_mode='r')
        ey = np.load(dataSourcePath+prefix_file_eti+str(nr_plikow)+files_postfix, mmap_mode='r')

        nx = (nx-mean)/odch_std  #PROBLEMY Z PAMIĘCCIĄ KOPIOWANIE TABLIC UKRYTE W SKRYPCIE

        X.append(nx)
        y_txt.append(ey)
        del nx
        del ey
        gc.collect()
      X = np.concatenate(X)
      y_txt = np.concatenate(y_txt)
      
      #X = np.transpose(X, (0,2,3,1))
      input_shape = X.shape[1:]   # Wymiary pojedynczego obrazu

    else:
      X = np.load(dataSourcePath+"abc_Dane32PrzetworzoneAVGlog.npy")
      y_txt = np.load(dataSourcePath+"abc_EtykietyDanychlog.npy", allow_pickle=True)
      X = np.transpose(X, (0,2,3,1))

      input_shape = X.shape[1:]   # Wymiary pojedynczego obrazu

    print(X.shape)
    print(y_txt.shape)
    print(input_shape)

    labelEnc = LabelEncoder()
    y = labelEnc.fit_transform(y_txt)
    class_amount = len(labelEnc.classes_)
    print(labelEnc.classes_)

    if not multifile_data:
      X_train_validate, X_test, y_train_validate, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
      X_train, X_validate, y_train, y_validate = train_test_split(X_train_validate, y_train_validate, test_size=0.25, random_state=42)

      #Normalizacja - standaryzacja (Z-score)
      mean = np.mean(X_train, axis=(0,1,2), keepdims=True) # 0-epoka, 1-czestotliwosc, 2-czas <-Do usrednienia     3-kanal
      std = np.std(X_train, axis=(0,1,2), keepdims=True )

      print(mean.shape)
      print("Srednie dla 21 kanalow: \n")
      print(mean[0][0][0])

      X_train= (X_train - mean)/std
      X_validate = (X_validate - mean)/std
      X_test = (X_test - mean)/std
    else:
      if if_test_run:
        X_test = X
        y_test = y
        
def train_model(multifile_data=False, first_training=True):
    global model,hist
    
    model = models.Sequential()
    model.add(layers.Conv2D(32, (1, 10), activation=None, padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())#  #
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2,2)))#   #
    model.add(layers.Conv2D(64, (3,1), activation=None, padding='same'))
    model.add(layers.BatchNormalization())#  #
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2,3)))
    model.add(layers.Dropout(0.6)) # (anti-overfitting)
    model.add(layers.DepthwiseConv2D(64, (3,3), activation=None, depth_multiplier=2, padding='same'))
    model.add(layers.BatchNormalization())#  #
    model.add(layers.Activation('relu'))
    model.add(layers.DepthwiseConv2D(64, (2,2), activation=None, padding='same'))
    model.add(layers.BatchNormalization())#  #
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2,2)))#   #

    model.add(layers.Flatten())
      #model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(64, activation=None))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.75)) # (anti-overfitting)
    model.add(layers.Dense(class_amount, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer = optimizer , loss = 'sparse_categorical_crossentropy' ,
                  metrics = ['accuracy'],#, metrics.Precision(), metrics.Recall(), metrics.AUC()],
                  jit_compile=False)
    model.summary()

    ReduceLROnPlateau_callback = callbacks.ReduceLROnPlateau(
          monitor='val_accuracy',
          patience = 5,
          verbose=1,
          factor=0.3,
          min_lr=0.0000001)

    EarlyStopping_callback = callbacks.EarlyStopping(
          monitor='val_loss',
          patience=10,
          start_from_epoch = 15,
          restore_best_weights=True,
          verbose=0,
          mode='min')

    if not first_training and multifile_data:
        model.load_weights(dataSourcePath+weight_file)
        print("Poprwanie wczytano wagi!")
    else:
        print("Utworzono nowy model!")

    epochs = 70
    batch_size=32  #16 # im mniejsza, tym większa dokładność, ale i więcej czasu

    if multifile_data:
      print("Poczatek treningu")
      hist = model.fit(
          x=X,
          y=y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=[ReduceLROnPlateau_callback, EarlyStopping_callback],
          #callbacks=[ReduceLROnPlateau_callback],
          shuffle=True
      )
      model.save_weights(dataSourcePath+weight_file)
      print("Trening zakonczony, model zapisany")
    else:
      print("Poczatek treningu (bez walidacji)")
      hist = model.fit(
          x=X_train,
          y=y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=[ReduceLROnPlateau_callback, EarlyStopping_callback],
          #callbacks=[ReduceLROnPlateau_callback],
          validation_data=(X_validate, y_validate),
          shuffle=True
      )

def test_model():
    global model,hist
    
    if len(X_test)>0 and len(y_test)>0:
      if not os.path.exists("data2/"+result_dir):
          os.mkdir("data2/"+result_dir)
        
      test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
      print(f"Skuteczność na danych testowych: {test_acc*100:.2f}%")

      y_predicted = model.predict(X_test)
      y_p_argmax = np.argmax(y_predicted, axis=1)
      #y_test_argmax = np.argmax(y_test.to_numpy(), axis=1)
      categories=["abstract","airplane","apple","banana","bird","boat","car","dog","person","train","zebra"]

      print(classification_report(y_test,y_p_argmax))
      matrix = confusion_matrix(y_test, y_p_argmax)

      disp = ConfusionMatrixDisplay(matrix, display_labels=categories)
      fig, ax = plt.subplots(figsize=(12,12))
      disp.plot(ax=ax, cmap="plasma") #viridis , plasma
      plt.xticks(rotation=45, ha='right')
      plt.savefig(f"{dataSourcePath}{result_dir}/MacierzPomyłek{now}.png", dpi=100, bbox_inches="tight")
      plt.close()

      wyk, (os1,os2) = plt.subplots(1,2, figsize=(20,5))
      os1.plot(hist.history['loss'],label='Strata',color='#AA0000',linewidth=2)
      if not multifile_data:
        os1.plot(hist.history['val_loss'],label='StrataVal',color='#FFAA00',linewidth=2)
      os1.set_title('Strata')
      os1.set_xlabel('Epoka',fontsize=10)
      os1.set_ylabel('Wartosc',fontsize=10)
      os1.grid(True,linestyle='--',alpha=0.6)
      os1.legend(fontsize=10)

      os2.plot(hist.history['accuracy'],label='Celność',color='#00AA00',linewidth=2)
      if not multifile_data:
        os2.plot(hist.history['val_accuracy'],label='CelnośćVal',color='#AAFF00',linewidth=2)
      os2.set_title('Celnosc')
      os2.set_xlabel('Epoka',fontsize=10)
      os2.set_ylabel('Wartosc',fontsize=10)
      os2.grid(True,linestyle='--',alpha=0.6)
      os2.legend(fontsize=10)
      
      plt.savefig(f"{dataSourcePath}{result_dir}/RecallAccuracy{now}.png", dpi=100, bbox_inches="tight")
      plt.close()

def save_model():
    global model
    if not os.path.exists("data2/"+result_dir):
        os.mkdir("data2/"+result_dir)
    model.save(dataSourcePath + "/" + result_dir + "/ModelCNN" + now + ".keras")

load_data(0, if_test_run=False)
train_model(multifile_data=True,first_training=True)

for i in range(1, train_file_no):
    load_data(i)
    train_model(multifile_data=True,first_training=False)

load_data(test_run, if_test_run=True)
test_model()

for i in range(0, train_file_no):
    load_data(i)
    train_model(multifile_data=True,first_training=False)
 
load_data(test_run, if_test_run=True)
test_model()
 
for i in range(0, train_file_no):
    load_data(i)
    train_model(multifile_data=True,first_training=False)

load_data(test_run, if_test_run=True)
test_model()

print("Trenowanie Zakonczono")
save_model()

#model.save(dataSourcePath + "Najlepszy3-09-0_44" + ".keras")
#zaladowany = keras.models.load_model(dataSourcePath+nazwa+".keras")