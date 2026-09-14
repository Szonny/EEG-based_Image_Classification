# -*- coding: utf-8 -*-
"""
Rekonstrukcja obrazow z sygnalu EEG (wariant A: autoenkoder EEG -> obraz).

Enkoder = korpus EEGNet (bez warstwy softmax) mapujacy surowy sygnal na wektor
cech (latent). Dekoder = sploty transponowane rozwijajace latent do obrazu RGB.
Uczenie: regresja do pikseli (strata MSE). Cel = rzeczywisty obraz ogladany
podczas danej proby.

UWAGA -- korzysta z plikow PER-BADANY z folderu data/ (nie z data2/), bo tylko
tam obraz (nazwa pliku) jest w tej samej kolejnosci co sygnal. Wymaga:
    data/{nick}_Dane32PrzetworzoneAVGRAW.npy      (sygnal surowy)
    data/{nick}_PlikiObrazow AVGRAW.npy           (nazwa pliku obrazu na probe)
oraz katalogu z obrazami bodzcow (IMAGES_DIR).
"""

import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
print("TensorFlow:", tf.__version__)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
                                      BatchNormalization, Activation, AveragePooling2D,
                                      Dropout, Flatten, Dense, Reshape, Conv2DTranspose)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split
from datetime import datetime

now = datetime.now().strftime("%m%d_%H%M")

# ============================ KONFIGURACJA ============================
dataSourcePath = "data/"          # pliki PER-BADANY (nie data2/)
NAZWA_DANE     = "_Dane32Przetworzone"
NAZWA_OBRAZY   = "_PlikiObrazow"
OZNACZENIE     = "AVGRAW"          # AVG + RAW

nicki_badanych = ["abc", "Bear", "fghx", "jt", "mi2", "miguel", "mole", "Reshi", "sapling"]

# Katalog z obrazami bodzcow. wczytaj_obraz() sklada sciezke -- DOSTOSUJ, jesli
# nazwy plikow w metadanych zawieraja juz podkatalog/rozszerzenie.
IMAGES_DIR = "images/"
IMG_SIZE   = 64                    # obrazy skalowane do IMG_SIZE x IMG_SIZE (potega 2 * 8)

DECYMACJA  = 4                     # jak w skrypcie EEGNet (~600 Hz -> ~150 Hz)
LATENT_DIM = 128                   # rozmiar wektora cech (wyjscie enkodera)
DROPOUT    = 0.25

BATCH_SIZE       = 64
EPOCHS           = 100
VALIDATION_SPLIT = 0.15
LEARNING_RATE    = 0.001
TEST_SIZE        = 0.2
RANDOM_STATE     = 42

folder_wynikowy = "rekonstrukcja"
# =====================================================================


# ------------------------ WCZYTYWANIE DANYCH ------------------------
def wczytaj_sygnal_i_identyfikatory():
    """Laczy pliki wszystkich badanych: sygnal + nazwa pliku obrazu (w tej samej
    kolejnosci). Zwraca (X, identyfikatory)."""
    X_parts, id_parts = [], []
    for nick in nicki_badanych:
        p_x  = f"{dataSourcePath}{nick}{NAZWA_DANE}{OZNACZENIE}.npy"
        p_id = f"{dataSourcePath}{nick}{NAZWA_OBRAZY}{OZNACZENIE}.npy"
        if not (os.path.exists(p_x) and os.path.exists(p_id)):
            raise FileNotFoundError(
                f"Brak plikow dla {nick}: {p_x} / {p_id}. Czy dodano zapis "
                f"'PlikiObrazow' w save_to_file i wygenerowano dane RAW?")
        X_parts.append(np.load(p_x))
        id_parts.append(np.load(p_id, allow_pickle=True))

    X = np.concatenate(X_parts).astype(np.float32)   # (N, kanaly, czas, 1)
    identyfikatory = np.concatenate(id_parts)
    del X_parts, id_parts
    gc.collect()

    # Decymacja w czasie (os 2) -- spojnie ze skryptem EEGNet.
    if DECYMACJA > 1:
        X = X[:, :, ::DECYMACJA, :]

    print(f"Sygnal: {X.shape}, identyfikatorow: {identyfikatory.shape}")
    return X, identyfikatory


def wczytaj_obraz(identyfikator):
    """Wczytuje i skaluje pojedynczy obraz bodzca. DOSTOSUJ sposob skladania
    sciezki do swojej struktury katalogow (np. podkatalog kategorii)."""
    sciezka = os.path.join(IMAGES_DIR, str(identyfikator))
    obraz = Image.open(sciezka).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(obraz, dtype=np.float32) / 255.0   # [0, 1]


def zbuduj_cele(identyfikatory):
    """Buduje tablice obrazow-celow (N, IMG_SIZE, IMG_SIZE, 3). Kazdy unikalny
    obraz wczytywany jest raz (cache), a nastepnie indeksowany na kazda probe."""
    unikalne = list(dict.fromkeys(identyfikatory.tolist()))
    cache = {u: wczytaj_obraz(u) for u in unikalne}
    print(f"Wczytano {len(unikalne)} unikalnych obrazow")
    Y = np.stack([cache[i] for i in identyfikatory.tolist()]).astype(np.float32)
    return Y


# ------------------------ MODEL ------------------------
def buduj_enkoder(Chans, Samples, latent_dim):
    """Korpus EEGNet zakonczony wektorem cech (bez softmax)."""
    inp = Input((Chans, Samples, 1))
    x = Conv2D(16, (1, min(64, Samples)), padding="same", use_bias=False)(inp)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)
    x = BatchNormalization()(x); x = Activation("elu")(x)
    x = AveragePooling2D((1, 4))(x); x = Dropout(DROPOUT)(x)
    x = SeparableConv2D(32, (1, 16), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x); x = Activation("elu")(x)
    x = AveragePooling2D((1, 8))(x); x = Dropout(DROPOUT)(x)
    x = Flatten()(x)
    latent = Dense(latent_dim, activation="elu", name="latent")(x)
    return Model(inp, latent, name="enkoder")


def buduj_dekoder(latent_dim, img_size=64, base=8):
    """Rozwija wektor cech do obrazu RGB przez sploty transponowane."""
    inp = Input((latent_dim,))
    x = Dense(base * base * 64, activation="elu")(inp)
    x = Reshape((base, base, 64))(x)
    n_up = int(np.log2(img_size // base))     # ile podwojen rozmiaru (8 -> img_size)
    filtry = 64
    for _ in range(n_up):
        filtry = max(filtry // 2, 16)
        x = Conv2DTranspose(filtry, 3, strides=2, padding="same", activation="elu")(x)
    out = Conv2D(3, 3, padding="same", activation="sigmoid")(x)   # [0,1], IMG_SIZE^2 x 3
    return Model(inp, out, name="dekoder")


def buduj_autoenkoder(Chans, Samples):
    enkoder = buduj_enkoder(Chans, Samples, LATENT_DIM)
    dekoder = buduj_dekoder(LATENT_DIM, IMG_SIZE)
    wejscie = Input((Chans, Samples, 1))
    wyjscie = dekoder(enkoder(wejscie))
    model = Model(wejscie, wyjscie, name="autoenkoder_eeg2obraz")
    model.compile(optimizer=Adam(LEARNING_RATE), loss="mse")
    model.summary()
    return model


# ------------------------ WIZUALIZACJA ------------------------
def zapisz_siatke(X_test, Y_test, model, ile=8):
    """Zapisuje siatke: gorny rzad = obraz prawdziwy, dolny = rekonstrukcja."""
    idx = np.random.choice(len(X_test), size=min(ile, len(X_test)), replace=False)
    rekonstrukcje = model.predict(X_test[idx])

    fig, osie = plt.subplots(2, len(idx), figsize=(2 * len(idx), 4.5))
    for k, i in enumerate(range(len(idx))):
        osie[0, k].imshow(np.clip(Y_test[idx[i]], 0, 1)); osie[0, k].axis("off")
        osie[1, k].imshow(np.clip(rekonstrukcje[i], 0, 1)); osie[1, k].axis("off")
    osie[0, 0].set_ylabel("Prawdziwy", fontsize=11)
    osie[1, 0].set_ylabel("Rekonstrukcja", fontsize=11)
    fig.suptitle("Obraz prawdziwy (gora) vs rekonstrukcja z EEG (dol)", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{dataSourcePath}{folder_wynikowy}/Rekonstrukcje{now}.png",
                dpi=120, bbox_inches="tight")
    plt.close()


# ------------------------ GLOWNA CZESC ------------------------
def main():
    os.makedirs(dataSourcePath + folder_wynikowy, exist_ok=True)

    X, identyfikatory = wczytaj_sygnal_i_identyfikatory()
    Y = zbuduj_cele(identyfikatory)

    # Standaryzacja sygnalu per kanal (statystyki wyznaczone na calosci -- dla
    # rekonstrukcji uproszczenie akceptowalne; mozna liczyc tylko na treningu).
    srednia = X.mean(axis=(0, 2, 3), keepdims=True)
    odch    = X.std(axis=(0, 2, 3), keepdims=True) + 1e-8
    X = (X - srednia) / odch

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
    del X, Y; gc.collect()

    Chans, Samples = X_train.shape[1], X_train.shape[2]
    model = buduj_autoenkoder(Chans, Samples)

    cb = [
        callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.3,
                                    min_lr=1e-7, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=15, start_from_epoch=15,
                                restore_best_weights=True, mode="min", verbose=1),
    ]

    print("Poczatek treningu")
    model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
              validation_split=VALIDATION_SPLIT, shuffle=True, callbacks=cb, verbose=1)

    # Ocena: MSE oraz SSIM (podobienstwo strukturalne) na zbiorze testowym.
    rek = model.predict(X_test, batch_size=BATCH_SIZE)
    mse  = float(np.mean((rek - Y_test) ** 2))
    ssim = float(tf.reduce_mean(tf.image.ssim(Y_test, rek, max_val=1.0)).numpy())
    print(f"Test MSE:  {mse:.4f}")
    print(f"Test SSIM: {ssim:.4f}  (1.0 = idealnie, ~0 = brak podobienstwa)")

    zapisz_siatke(X_test, Y_test, model, ile=8)
    model.save(f"{dataSourcePath}{folder_wynikowy}/Autoenkoder{now}.keras")
    print("Zapisano model i siatke rekonstrukcji.")


if __name__ == "__main__":
    main()
