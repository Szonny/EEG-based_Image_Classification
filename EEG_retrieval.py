# -*- coding: utf-8 -*-
"""
Rekonstrukcja obrazow z EEG -- wariant B: identyfikacja / retrieval.

Zamiast generowac piksele, uczymy enkoder EEG -> embedding (wektor cech obrazu),
a nastepnie sprawdzamy, czy embedding z EEG trafia blizej WLASCIWEGO obrazu niz
obrazow losowych. To omija problem "ucieczki w srednia" ze straty MSE i daje
konkretne metryki:
  * retrieval top-1 / top-5 (wsrod wszystkich unikalnych obrazow),
  * identyfikacja n-way (czy poprawny obraz wygrywa z k-1 losowymi rozpraszaczami).

Embedding obrazu-celu liczymy pretrenowana siecia MobileNetV2 (semantyczny)
albo -- bez pobierania wag -- z pomniejszonych pikseli (TRYB_OSADZEN="piksele").

Wymaga tych samych plikow co skrypt rekonstrukcji (dane RAW per-badany + PlikiObrazow).
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
                                      Dropout, Flatten, Dense, Lambda)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CosineSimilarity
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split
from datetime import datetime

now = datetime.now().strftime("%m%d_%H%M")

# ============================ KONFIGURACJA ============================
dataSourcePath = "data/"
NAZWA_DANE     = "_Dane32Przetworzone"
NAZWA_OBRAZY   = "_PlikiObrazow"
OZNACZENIE     = "AVGRAW"

nicki_badanych = ["abc", "Bear", "fghx", "jt", "mi2", "miguel", "mole", "Reshi", "sapling"]

IMAGES_DIR = "images/"       # folder nadrzedny z podkatalogami kategorii (jak w rekonstrukcji)
IMG_SIZE   = 64              # rozmiar obrazu podawanego do enkodera obrazow
PIXEL_SIZE = 16             # rozmiar dla embeddingu "piksele" (16x16x3 = 768)

# Zrodlo embeddingu obrazu-celu:
#   "pretrenowany" -> MobileNetV2 (semantyczny; przy 1. uruchomieniu pobiera wagi z internetu)
#   "piksele"      -> pomniejszone piksele (bez pobierania niczego)
TRYB_OSADZEN = "pretrenowany"

DECYMACJA  = 4
DROPOUT    = 0.25

BATCH_SIZE       = 64
EPOCHS           = 100
VALIDATION_SPLIT = 0.15
LEARNING_RATE    = 0.001
TEST_SIZE        = 0.2
RANDOM_STATE     = 42

# Ewaluacja n-way
N_WAY   = (2, 5, 10)         # ile obrazow "do wyboru" (1 poprawny + k-1 rozpraszaczy)
POWTORZ = 100               # ile losowan rozpraszaczy na probe

# Kategoria pomijana w analizie "tylko zdjecia" (aby sprawdzic, czy sygnal siega
# poza latwo odrozialny abstrakt). Kategorie odczytujemy z przedrostka sciezki
# w nazwie pliku obrazu (np. "abstract/..." -> "abstract").
KATEGORIA_WYKLUCZ = "abstract"

folder_wynikowy = "rekonstrukcja"
# =====================================================================


# ------------------------ WCZYTYWANIE ------------------------
def wczytaj_obraz(nazwa, rozmiar):
    sciezka = os.path.join(IMAGES_DIR, str(nazwa))
    obraz = Image.open(sciezka).convert("RGB").resize((rozmiar, rozmiar))
    return np.asarray(obraz, dtype=np.float32) / 255.0


def wczytaj_sygnal_i_identyfikatory():
    X_parts, id_parts = [], []
    for nick in nicki_badanych:
        p_x  = f"{dataSourcePath}{nick}{NAZWA_DANE}{OZNACZENIE}.npy"
        p_id = f"{dataSourcePath}{nick}{NAZWA_OBRAZY}{OZNACZENIE}.npy"
        if not (os.path.exists(p_x) and os.path.exists(p_id)):
            raise FileNotFoundError(f"Brak plikow dla {nick}: {p_x} / {p_id}.")
        X_parts.append(np.load(p_x))
        id_parts.append(np.load(p_id, allow_pickle=True))

    X = np.concatenate(X_parts).astype(np.float32)          # (N, kanaly, czas, 1)
    identyfikatory = np.concatenate(id_parts).astype(str)
    del X_parts, id_parts; gc.collect()

    if DECYMACJA > 1:
        X = X[:, :, ::DECYMACJA, :]

    # Standaryzacja per kanal
    srednia = X.mean(axis=(0, 2, 3), keepdims=True)
    odch    = X.std(axis=(0, 2, 3), keepdims=True) + 1e-8
    X = (X - srednia) / odch

    print(f"Sygnal: {X.shape}, identyfikatorow: {identyfikatory.shape}")
    return X, identyfikatory


def osadz_obrazy(unikalne_pliki):
    """Zwraca (M, D) embeddingow obrazow (znormalizowane L2)."""
    if TRYB_OSADZEN == "pretrenowany":
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        obrazy = np.stack([wczytaj_obraz(f, IMG_SIZE) for f in unikalne_pliki])
        baza = MobileNetV2(include_top=False, weights="imagenet", pooling="avg",
                           input_shape=(IMG_SIZE, IMG_SIZE, 3))
        emb = baza.predict(preprocess_input(obrazy * 255.0), batch_size=32, verbose=0)
    else:  # "piksele"
        maly = np.stack([wczytaj_obraz(f, PIXEL_SIZE) for f in unikalne_pliki])
        emb = maly.reshape(len(unikalne_pliki), -1)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    print(f"Embedding obrazow: {emb.shape} (tryb: {TRYB_OSADZEN})")
    return emb.astype(np.float32)


# ------------------------ ENKODER EEG ------------------------
def buduj_enkoder_eeg(Chans, Samples, embed_dim):
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
    x = Dense(256, activation="elu")(x)
    x = Dropout(DROPOUT)(x)
    x = Dense(embed_dim)(x)
    out = Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)   # embedding na sferze
    model = Model(inp, out, name="enkoder_eeg")
    model.compile(optimizer=Adam(LEARNING_RATE), loss=CosineSimilarity(axis=1))
    model.summary()
    return model


# ------------------------ METRYKI ------------------------
def n_way_identyfikacja(sims, true_idx, k, powtorz, seed=0):
    """Srednia trafnosc: czy poprawny obraz ma wyzsze podobienstwo niz k-1 losowych."""
    rng = np.random.default_rng(seed)
    Nt, M = sims.shape
    wygrane, razem = 0, Nt * powtorz
    for i in range(Nt):
        ti = int(true_idx[i])
        cs = sims[i, ti]
        inne = sims[i][np.arange(M) != ti]              # (M-1,)
        idx = rng.integers(0, M - 1, size=(powtorz, k - 1))
        maxd = inne[idx].max(axis=1)                     # najlepszy rozpraszacz w kazdym losowaniu
        wygrane += int(np.sum(cs > maxd))
    return wygrane / razem


def metryki(sims, true_idx, etykieta):
    """Wypisuje top-1, top-5 oraz n-way dla podanego zbioru podobienstw."""
    Nt, M = sims.shape
    if Nt == 0:
        print(f"\n--- {etykieta}: brak prob do oceny ---")
        return
    rank = np.argsort(-sims, axis=1)
    top1 = float(np.mean(rank[:, 0] == true_idx))
    top5 = float(np.mean([true_idx[i] in rank[i, :5] for i in range(Nt)]))
    print(f"\n--- {etykieta} (kandydatow: {M}, prob testowych: {Nt}) ---")
    print(f"top-1: {top1*100:.2f}%   (losowo ~ {100.0/M:.2f}%)")
    print(f"top-5: {top5*100:.2f}%   (losowo ~ {min(500.0/M, 100.0):.2f}%)")
    for k in N_WAY:
        if k <= M:
            acc = n_way_identyfikacja(sims, true_idx, k, POWTORZ)
            print(f"{k}-way: {acc*100:.2f}%   (losowo {100.0/k:.1f}%)")


def zapisz_przyklady(sims, true_idx, unikalne_pliki, ile=5, topn=5):
    rank = np.argsort(-sims, axis=1)
    wybrane = np.random.choice(len(sims), size=min(ile, len(sims)), replace=False)
    fig, ax = plt.subplots(len(wybrane), topn + 1, figsize=(2 * (topn + 1), 2 * len(wybrane)))
    if len(wybrane) == 1:
        ax = ax[None, :]
    for r, i in enumerate(wybrane):
        ax[r, 0].imshow(wczytaj_obraz(unikalne_pliki[int(true_idx[i])], IMG_SIZE))
        ax[r, 0].axis("off"); ax[r, 0].set_title("prawdziwy", fontsize=8)
        for c in range(topn):
            ax[r, c + 1].imshow(wczytaj_obraz(unikalne_pliki[rank[i, c]], IMG_SIZE))
            ax[r, c + 1].axis("off"); ax[r, c + 1].set_title(f"#{c+1}", fontsize=8)
    fig.suptitle("Zapytanie (obraz prawdziwy) vs najblizsze obrazy wg embeddingu z EEG", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{dataSourcePath}{folder_wynikowy}/Retrieval{now}.png", dpi=120, bbox_inches="tight")
    plt.close()


# ------------------------ GLOWNA CZESC ------------------------
def main():
    os.makedirs(dataSourcePath + folder_wynikowy, exist_ok=True)

    X, identyfikatory = wczytaj_sygnal_i_identyfikatory()

    # Unikalne obrazy + ich embeddingi; mapowanie proba -> wiersz embeddingu
    unikalne_pliki = list(dict.fromkeys(identyfikatory.tolist()))
    plik_do_wiersza = {f: i for i, f in enumerate(unikalne_pliki)}
    emb_obrazow = osadz_obrazy(unikalne_pliki)                       # (M, D)
    wiersze = np.array([plik_do_wiersza[f] for f in identyfikatory.tolist()])
    Y = emb_obrazow[wiersze]                                          # cel treningowy (N, D)

    X_tr, X_te, Y_tr, Y_te, w_tr, w_te = train_test_split(
        X, Y, wiersze, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
    del X, Y; gc.collect()

    Chans, Samples = X_tr.shape[1], X_tr.shape[2]
    model = buduj_enkoder_eeg(Chans, Samples, emb_obrazow.shape[1])

    cb = [
        callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.3,
                                    min_lr=1e-7, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=15, start_from_epoch=15,
                                restore_best_weights=True, mode="min", verbose=1),
    ]
    print("Poczatek treningu")
    model.fit(X_tr, Y_tr, batch_size=BATCH_SIZE, epochs=EPOCHS,
              validation_split=VALIDATION_SPLIT, shuffle=True, callbacks=cb, verbose=1)

    # --- Ewaluacja ---
    pred = model.predict(X_te, batch_size=BATCH_SIZE)
    pred = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
    sims = pred @ emb_obrazow.T                                       # (Nt, M)

    print("\n================= WYNIKI (retrieval / identyfikacja) =================")

    # (1) Wszystkie kategorie
    metryki(sims, w_te, "Wszystkie kategorie")

    # (2) Tylko zdjecia -- z pominieciem latwo odrozialnego abstraktu.
    #     Kategorie odczytujemy z przedrostka sciezki (np. "abstract/...").
    kategorie = np.array([str(f).replace("\\", "/").split("/")[0] for f in unikalne_pliki])
    is_wykluczona = (kategorie == KATEGORIA_WYKLUCZ)
    foto_cols = np.where(~is_wykluczona)[0]                           # indeksy zdjec (pelna lista)
    n_wyklucz = int(is_wykluczona.sum())

    if 0 < len(foto_cols) < len(kategorie):
        col_map = {full: j for j, full in enumerate(foto_cols)}
        te_foto = np.where(~is_wykluczona[w_te])[0]                   # proby, ktorych cel jest zdjeciem
        sims_foto = sims[np.ix_(te_foto, foto_cols)]                  # (len(te_foto), M_foto)
        true_foto = np.array([col_map[w_te[i]] for i in te_foto])
        print(f"\n(pominieto kategorie '{KATEGORIA_WYKLUCZ}': {n_wyklucz} obrazow)")
        metryki(sims_foto, true_foto, f"Tylko zdjecia (bez '{KATEGORIA_WYKLUCZ}')")
    else:
        print(f"\n(nie znaleziono kategorii '{KATEGORIA_WYKLUCZ}' do pominiecia -- "
              f"analiza 'tylko zdjecia' pominieta)")

    print("======================================================================\n")

    zapisz_przyklady(sims, w_te, unikalne_pliki, ile=5, topn=5)
    print(f"Zapisano przyklady retrieval w {dataSourcePath}{folder_wynikowy}/Retrieval{now}.png")


if __name__ == "__main__":
    main()