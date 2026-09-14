# -*- coding: utf-8 -*-
"""
Uczenie i klasyfikacja obrazow z sygnalu EEG za pomoca sieci EEGNet.

EEGNet (Lawhern i in., 2018) to kompaktowa siec CNN zaprojektowana dla
SUROWEGO sygnalu EEG. Oczekuje danych w postaci (kanaly, probki_czasowe, 1),
dlatego korzysta z surowych epok (wariant do_morlet=False w potoku
przygotowania danych), a NIE ze spektrogramow.

Wymagane pliki (wygenerowane przez data_preparation + data_randomizer w wariancie
RAW, tzn. do_morlet=False / do_TFA=False):
    data2/plikWynikowy{N}_TRAIN_Dane32PrzetworzoneAVGRAW.npy
    data2/plikWynikowyET{N}_TRAIN_Dane32PrzetworzoneAVGRAW.npy
    data2/plikWynikowy{N}_TEST_Dane32PrzetworzoneAVGRAW.npy
    data2/plikWynikowyET{N}_TEST_Dane32PrzetworzoneAVGRAW.npy
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
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
                                      BatchNormalization, Activation, AveragePooling2D,
                                      Dropout, Flatten, Dense)
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

folder_wynikowy = "finito_eegnet"
sciezka_wag     = "eegnet.weights.h5"

categories = ["abstract", "airplane", "apple", "banana", "bird", "boat",
              "car", "dog", "person", "train", "zebra"]

# --- Przetwarzanie sygnalu ---
# Decymacja (podprobkowanie) sygnalu w czasie. Dane sa odfiltrowane do <=40 Hz,
# a probkowane ~600 Hz (301 probek / 0,5 s) -- czyli mocno nadprobkowane.
# DECYMACJA=4 -> ~150 Hz (~76 probek): usuwa redundancje i dopasowuje dlugosc
# jadra czasowego do sygnalu. Ustaw 1, aby wylaczyc.
DECYMACJA = 4

# --- Hiperparametry EEGNet ---
F1          = 16         # liczba filtrow czasowych                 (poprzednio: 8)
D           = 2          # mnoznik glebokosci (filtry przestrzenne na filtr czasowy)
F2          = F1 * D     # liczba filtrow w splocie separowalnym     (= 32)
DROPOUT     = 0.25       # zmniejszone z 0.5 -- model sie niedouczal (poprzednio: 0.5)
KERN_LENGTH = 64         # ~polowa czestotliwosci probkowania PO decymacji (~150 Hz -> ~75)

# --- Uczenie ---
BATCH_SIZE       = 64
EPOCHS           = 100
VALIDATION_SPLIT = 0.15
LEARNING_RATE    = 0.001

# Tryb: "trenuj" -> ucz + zapisz + testuj;  "testuj" -> tylko wczytaj wagi i testuj
TRYB = "trenuj"

# Sposob podzialu danych na trening/test:
#   "miedzyosobowy" -> TRAIN = badani 0-6, TEST = badani 7-8 (osoby spoza treningu);
#                      trudniejszy scenariusz, mierzy generalizacje na NOWE osoby.
#   "mieszany"      -> proby wszystkich badanych laczone, tasowane i dzielone losowo;
#                      latwiejszy scenariusz (te same osoby w treningu i w tescie).
TRYB_PODZIALU = "mieszany"
TEST_SIZE     = 0.2       # udzial zbioru testowego w trybie "mieszany"
RANDOM_STATE  = 42
# =====================================================================

# Enkoder etykiet dopasowany na stalej liscie kategorii -> deterministyczna,
# spojna kolejnosc klas w treningu, tescie i na macierzy pomylek.
labelEnc = LabelEncoder().fit(categories)


def EEGNet(nb_classes, Chans, Samples, dropoutRate=0.5, kernLength=64,
           F1=8, D=2, F2=16, norm_rate=0.25):
    """Kanoniczna architektura EEGNet (wariant v2, Lawhern i in. 2018)."""
    wejscie = Input(shape=(Chans, Samples, 1))

    # --- Blok 1: splot czasowy + splot przestrzenny (depthwise) ---
    x = Conv2D(F1, (1, kernLength), padding="same", use_bias=False)(wejscie)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D,
                        depthwise_constraint=max_norm(1.0))(x)   # filtruje po kanalach
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = AveragePooling2D((1, 4))(x)
    x = Dropout(dropoutRate)(x)

    # --- Blok 2: splot separowalny (podsumowanie czasowe) ---
    x = SeparableConv2D(F2, (1, 16), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = AveragePooling2D((1, 8))(x)
    x = Dropout(dropoutRate)(x)

    x = Flatten()(x)
    x = Dense(nb_classes, kernel_constraint=max_norm(norm_rate))(x)
    wyjscie = Activation("softmax")(x)
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

    Dane z data_randomizer sa JUZ znormalizowane, dlatego NIE normalizujemy ich
    ponownie (to unika podwojnej standaryzacji obecnej w poprzednim skrypcie).
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
    # EEGNet oczekuje (N, kanaly, czas, 1) -> przestawiamy osie.
    X = np.transpose(X, (0, 3, 1, 2))
    assert X.shape[-1] == 1, (
        f"Po przestawieniu osi ostatni wymiar powinien wynosic 1 (dane surowe), "
        f"a otrzymano {X.shape}. Czy na pewno uzywasz danych RAW (do_morlet=False), "
        f"a nie spektrogramow? Jesli tak, dostosuj transpozycje w wczytaj_zbior().")

    # Decymacja w czasie (os 2) -- zmniejsza nadprobkowanie; patrz DECYMACJA w konfiguracji.
    if DECYMACJA > 1:
        X = X[:, :, ::DECYMACJA, :]

    print(f"[{etap}] dane: {X.shape}, etykiety: {y_txt.shape}")
    return X, y_txt


def zbuduj_i_skompiluj(input_shape, nb_classes):
    Chans, Samples = input_shape[0], input_shape[1]
    model = EEGNet(nb_classes=nb_classes, Chans=Chans, Samples=Samples,
                   dropoutRate=DROPOUT, kernLength=min(KERN_LENGTH, Samples),
                   F1=F1, D=D, F2=F2)
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
    """Zwraca (X_train, y_train, X_test, y_test) zgodnie z TRYB_PODZIALU.

    "miedzyosobowy": TRAIN i TEST to gotowe, rozlaczne wzgledem badanych zbiory
                     z data_randomizer (TEST = osoby spoza treningu).
    "mieszany":      proby ze WSZYSTKICH badanych (pliki TRAIN + TEST) sa laczone,
                     tasowane i dzielone losowo z zachowaniem proporcji klas.
    """
    if TRYB_PODZIALU == "miedzyosobowy":
        X_train, y_train_txt = wczytaj_zbior("TRAIN")
        X_test,  y_test_txt  = wczytaj_zbior("TEST")
        y_train = labelEnc.transform(y_train_txt)
        y_test  = labelEnc.transform(y_test_txt)

    elif TRYB_PODZIALU == "mieszany":
        # Pliki TRAIN (badani 0-6) + TEST (badani 7-8) razem daja proby wszystkich
        # badanych. Uwaga: obie czesci byly znormalizowane wlasnymi statystykami,
        # co przy laczeniu wprowadza drobna niespojnosc (pomijalna, bo dane sa juz
        # w przyblizeniu wystandaryzowane).
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
        model.save(f"{dataSourcePath}{folder_wynikowy}/EEGNet{now}.keras")
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