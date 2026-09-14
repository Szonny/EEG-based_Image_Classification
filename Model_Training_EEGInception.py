# -*- coding: utf-8 -*-
"""
Uczenie i klasyfikacja obrazow z sygnalu EEG za pomoca sieci EEG-Inception.

EEG-Inception (Santamaria-Vazquez i in., 2020) to kompaktowa siec CNN dla
SUROWEGO sygnalu EEG, wykorzystujaca moduly "inception": rownolegle sploty
czasowe o roznych skalach (wielorozdzielczosc), a nastepnie splot przestrzenny
typu depthwise po kanalach. Oczekuje danych w postaci (probki_czasowe, kanaly, 1)
-- czyli ODWROTNIE niz EEGNet (kanaly, probki_czasowe, 1).

Wymagane pliki: te same co dla skryptu EEGNet (wariant RAW, do_morlet=False):
    data2/plikWynikowy{N}_TRAIN_Dane32PrzetworzoneAVGRAW.npy   (itd. + ET, TEST)
"""

import os
import re
import gc
import glob
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
print("TensorFlow:", tf.__version__)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D, BatchNormalization,
                                      Activation, AveragePooling2D, Dropout, Flatten,
                                      Dense, concatenate)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime

now = datetime.now().strftime("%m%d_%H%M")

# ============================ KONFIGURACJA ============================
dataSourcePath   = "data2/"
PRZEDROSTEK_DANE = "plikWynikowy"
PRZEDROSTEK_ETYK = "plikWynikowyET"
NAZWA_PLIKU      = "_Dane32Przetworzone"

# Dla danych SUROWYCH: US_SREDNIA in {"AVG", "NoAVG"}, REDUKCJA = "RAW"
US_SREDNIA = "AVG"
REDUKCJA   = "RAW"
OZNACZENIE = US_SREDNIA + REDUKCJA            # np. "AVGRAW"

folder_wynikowy = "finito_eeginception"
sciezka_wag     = "eeginception.weights.h5"

categories = ["abstract", "airplane", "apple", "banana", "bird", "boat",
              "car", "dog", "person", "train", "zebra"]

# --- Przetwarzanie sygnalu ---
# Decymacja w czasie (jak w skrypcie EEGNet). DECYMACJA=4 -> ~150 Hz (~76 probek).
# Ustaw 1, aby wylaczyc.
DECYMACJA = 4

# --- Hiperparametry EEG-Inception ---
FILTRY_NA_GALAZ  = 8          # liczba filtrow na kazda galaz modulu inception
DEPTH_MULTIPLIER = 2          # mnoznik glebokosci splotu przestrzennego (po kanalach)
DROPOUT          = 0.25
# Skale czasowe modulu inception = Samples // dzielnik (rozne dlugosci jader ->
# wielorozdzielczosc). Wyznaczane automatycznie z dlugosci sygnalu.
SCALE_DZIELNIKI  = (2, 4, 8)

# --- Uczenie ---
BATCH_SIZE       = 64
EPOCHS           = 100
VALIDATION_SPLIT = 0.15
LEARNING_RATE    = 0.001

# Tryb: "trenuj" -> ucz + zapisz + testuj;  "testuj" -> tylko wczytaj wagi i testuj
TRYB = "trenuj"

# Sposob podzialu danych na trening/test (jak w skrypcie EEGNet):
#   "miedzyosobowy" -> TEST = osoby spoza treningu (trudniejsze).
#   "mieszany"      -> proby wszystkich badanych, dzielone losowo (latwiejsze).
TRYB_PODZIALU = "mieszany"
TEST_SIZE     = 0.2
RANDOM_STATE  = 42
# =====================================================================

labelEnc = LabelEncoder().fit(categories)


def EEGInception(nb_classes, Samples, Chans, filters_per_branch=8,
                 scales_samples=(32, 16, 8), dropout_rate=0.25,
                 activation="elu", depth_multiplier=2):
    """Architektura EEG-Inception (Santamaria-Vazquez i in. 2020)."""
    wejscie = Input(shape=(Samples, Chans, 1))

    # ===== Blok 1: modul inception (sploty czasowe + splot przestrzenny) =====
    galezie_1 = []
    for skala in scales_samples:
        u = Conv2D(filters_per_branch, (skala, 1), padding="same")(wejscie)
        u = BatchNormalization()(u)
        u = Activation(activation)(u)
        u = Dropout(dropout_rate)(u)
        u = DepthwiseConv2D((1, Chans), use_bias=False, depth_multiplier=depth_multiplier,
                            depthwise_constraint=max_norm(1.0))(u)   # filtruje po kanalach
        u = BatchNormalization()(u)
        u = Activation(activation)(u)
        u = Dropout(dropout_rate)(u)
        galezie_1.append(u)
    x = concatenate(galezie_1, axis=3)
    x = AveragePooling2D((4, 1))(x)

    # ===== Blok 2: modul inception (krotsze skale) =====
    galezie_2 = []
    for skala in scales_samples:
        u = Conv2D(filters_per_branch, (max(skala // 4, 1), 1), padding="same")(x)
        u = BatchNormalization()(u)
        u = Activation(activation)(u)
        u = Dropout(dropout_rate)(u)
        galezie_2.append(u)
    x = concatenate(galezie_2, axis=3)
    x = AveragePooling2D((2, 1))(x)

    # ===== Blok 3: warstwy wyjsciowe =====
    n = filters_per_branch * len(scales_samples)
    x = Conv2D(n // 2, (8, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = AveragePooling2D((2, 1))(x)
    x = Dropout(dropout_rate)(x)

    x = Conv2D(n // 4, (4, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = AveragePooling2D((2, 1))(x)
    x = Dropout(dropout_rate)(x)

    x = Flatten()(x)
    wyjscie = Dense(nb_classes, activation="softmax")(x)
    return Model(inputs=wejscie, outputs=wyjscie)


def _pliki_zbioru(etap):
    """Zwraca posortowana liste plikow danych dla etapu TRAIN/TEST."""
    wzor = f"{dataSourcePath}{PRZEDROSTEK_DANE}[0-9]*_{etap}{NAZWA_PLIKU}{OZNACZENIE}.npy"

    def numer(p):
        m = re.search(rf"{PRZEDROSTEK_DANE}(\d+)_", os.path.basename(p))
        return int(m.group(1)) if m else -1

    return sorted(glob.glob(wzor), key=numer)


def wczytaj_zbior(etap):
    """Wczytuje wszystkie pliki danego etapu i skleja w jedna tablice.
    Dane sa juz znormalizowane przez data_randomizer -- nie normalizujemy ponownie.
    """
    pliki = _pliki_zbioru(etap)
    if not pliki:
        raise FileNotFoundError(
            f"Nie znaleziono plikow dla etapu {etap} (oznaczenie {OZNACZENIE}) "
            f"w katalogu {dataSourcePath}. Czy dane RAW zostaly wygenerowane?")

    X_parts, y_parts = [], []
    for p in pliki:
        p_et = os.path.basename(p).replace(PRZEDROSTEK_DANE, PRZEDROSTEK_ETYK, 1)
        p_et = os.path.join(os.path.dirname(p), p_et)
        X_parts.append(np.load(p))
        y_parts.append(np.load(p_et, allow_pickle=True))

    X = np.concatenate(X_parts).astype(np.float32)
    y_txt = np.concatenate(y_parts)
    del X_parts, y_parts
    gc.collect()

    # data_randomizer zapisuje dane SUROWE w ukladzie (N, czas, 1, kanaly).
    # EEG-Inception oczekuje (N, czas, kanaly, 1) -> przestawiamy osie.
    X = np.transpose(X, (0, 1, 3, 2))
    assert X.shape[-1] == 1, (
        f"Po przestawieniu osi ostatni wymiar powinien wynosic 1 (dane surowe), "
        f"a otrzymano {X.shape}. Czy na pewno uzywasz danych RAW (do_morlet=False)?")

    # Decymacja w czasie (tutaj os 1) -- patrz DECYMACJA w konfiguracji.
    if DECYMACJA > 1:
        X = X[:, ::DECYMACJA, :, :]

    print(f"[{etap}] dane: {X.shape}, etykiety: {y_txt.shape}")
    return X, y_txt


def zbuduj_i_skompiluj(input_shape, nb_classes):
    Samples, Chans = input_shape[0], input_shape[1]
    scales = tuple(max(Samples // d, 1) for d in SCALE_DZIELNIKI)
    model = EEGInception(nb_classes=nb_classes, Samples=Samples, Chans=Chans,
                         filters_per_branch=FILTRY_NA_GALAZ, scales_samples=scales,
                         dropout_rate=DROPOUT, depth_multiplier=DEPTH_MULTIPLIER)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()
    return model


def rysuj_wyniki(model, hist, X_test, y_test):
    os.makedirs(dataSourcePath + folder_wynikowy, exist_ok=True)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Skutecznosc na danych testowych: {test_acc * 100:.2f}%")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    etykiety_klas = list(labelEnc.classes_)
    print(classification_report(y_test, y_pred, target_names=etykiety_klas))

    macierz = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(macierz, display_labels=etykiety_klas)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(ax=ax, cmap="plasma")
    plt.xticks(rotation=45, ha="right")
    plt.savefig(f"{dataSourcePath}{folder_wynikowy}/MacierzPomylek{now}.png",
                dpi=100, bbox_inches="tight")
    plt.close()

    fig, (o1, o2) = plt.subplots(1, 2, figsize=(20, 5))
    o1.plot(hist.history["loss"], label="Strata", color="#AA0000", linewidth=2)
    if "val_loss" in hist.history:
        o1.plot(hist.history["val_loss"], label="Strata (walid.)", color="#FFAA00", linewidth=2)
    o1.set_title("Strata"); o1.set_xlabel("Epoka"); o1.set_ylabel("Wartosc")
    o1.grid(True, linestyle="--", alpha=0.6); o1.legend()

    o2.plot(hist.history["accuracy"], label="Celnosc", color="#00AA00", linewidth=2)
    if "val_accuracy" in hist.history:
        o2.plot(hist.history["val_accuracy"], label="Celnosc (walid.)", color="#AAFF00", linewidth=2)
    o2.set_title("Celnosc"); o2.set_xlabel("Epoka"); o2.set_ylabel("Wartosc")
    o2.grid(True, linestyle="--", alpha=0.6); o2.legend()

    plt.savefig(f"{dataSourcePath}{folder_wynikowy}/KrzyweUczenia{now}.png",
                dpi=100, bbox_inches="tight")
    plt.close()


def przygotuj_zbiory():
    """Zwraca (X_train, y_train, X_test, y_test) zgodnie z TRYB_PODZIALU."""
    if TRYB_PODZIALU == "miedzyosobowy":
        X_train, y_train_txt = wczytaj_zbior("TRAIN")
        X_test,  y_test_txt  = wczytaj_zbior("TEST")
        y_train = labelEnc.transform(y_train_txt)
        y_test  = labelEnc.transform(y_test_txt)

    elif TRYB_PODZIALU == "mieszany":
        Xa, ya = wczytaj_zbior("TRAIN")
        Xb, yb = wczytaj_zbior("TEST")
        X_all = np.concatenate([Xa, Xb]); del Xa, Xb; gc.collect()
        y_all = labelEnc.transform(np.concatenate([ya, yb]))
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            shuffle=True, stratify=y_all)
        del X_all, y_all; gc.collect()
        print(f"[mieszany] train: {X_train.shape}, test: {X_test.shape}")

    else:
        raise ValueError(f"Nieznany TRYB_PODZIALU: {TRYB_PODZIALU} "
                         f"(uzyj 'miedzyosobowy' lub 'mieszany').")

    return X_train, y_train, X_test, y_test


def main():
    os.makedirs(dataSourcePath + folder_wynikowy, exist_ok=True)
    print(f"Tryb podzialu danych: {TRYB_PODZIALU}")

    X_train, y_train, X_test, y_test = przygotuj_zbiory()

    if TRYB == "trenuj":
        model = zbuduj_i_skompiluj(X_train.shape[1:], nb_classes=len(categories))

        cb = [
            callbacks.ReduceLROnPlateau(monitor="val_accuracy", patience=5,
                                        factor=0.3, min_lr=1e-7, verbose=1),
            callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                    start_from_epoch=15, restore_best_weights=True,
                                    mode="min", verbose=1),
        ]

        print("Poczatek treningu")
        hist = model.fit(X_train, y_train,
                         batch_size=BATCH_SIZE, epochs=EPOCHS,
                         validation_split=VALIDATION_SPLIT, shuffle=True,
                         callbacks=cb, verbose=1)

        model.save_weights(dataSourcePath + sciezka_wag)
        model.save(f"{dataSourcePath}{folder_wynikowy}/EEGInception{now}.keras")
        print("Trening zakonczony, model zapisany")

        rysuj_wyniki(model, hist, X_test, y_test)

    else:  # TRYB == "testuj"
        model = zbuduj_i_skompiluj(X_test.shape[1:], nb_classes=len(categories))
        model.load_weights(dataSourcePath + sciezka_wag)
        print("Wczytano wagi modelu")

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Skutecznosc na danych testowych: {test_acc * 100:.2f}%")
        y_pred = np.argmax(model.predict(X_test), axis=1)
        print(classification_report(y_test, y_pred, target_names=list(labelEnc.classes_)))

    print("Zakonczono.")


if __name__ == "__main__":
    main()