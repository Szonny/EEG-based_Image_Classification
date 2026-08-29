import pandas  as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import mne
import sklearn

import io
import imageio.v3 as iio
import matplotlib.pyplot as plt
from datetime import datetime
now = datetime.now().strftime("%m%d_%H%M")

nicki_badanych = ["abc", "Bear", "fghx", "jt", "mi2", "miguel", "mole", "Reshi", "sapling"]
csv_eventow = ["abc_EEGBasedVisualRecall_Events_Rep1_2026-05-27_11-04-56",
               "Bear_EEGBasedVisualRecall_Events_Rep1_2026-05-29_10-34-33",
               "fghx_EEGBasedVisualRecall_Events_Rep1_2026-05-27_14-21-40",
               "jt_EEGBasedVisualRecall_Events_Rep1_2026-06-10_10-07-50",
               "MI2_EEGBasedVisualRecall_Events_Rep1_2026-06-18_11-42-39",
               "miguel_EEGBasedVisualRecall_Events_Rep1_2026-06-11_09-42-59",
               "mole_EEGBasedVisualRecall_Events_Rep2_2026-05-22_11-27-28",
               "Reshi_EEGBasedVisualRecall_Events_Rep1_2026-05-28_09-40-36",
               "sapling_EEGBasedVisualRecall_Events_Rep1_2026-06-12_09-43-56"]
reduction_methods = ["ratio", "logratio", "zlogratio", "mean", "zscore"]

# Wybór obecnych badanych
wybrani_badani = [0,5]

obecnie_badani = [nicki_badanych[i] for i in wybrani_badani]
NICK_BADANEGO = nicki_badanych[wybrani_badani[0]]
CSV_EVENTY =    csv_eventow[wybrani_badani[0]]

# Opcje edycji epok
subtract_mean_baseline = False

reduct_tfa_using_baseline = False
chosen_reduction_method = reduction_methods[2]

# Opcje zapisu do .npy
do_save_to_file = True

# Opcje wizualizacji
visualization_catalogue="vis"
image_channel = 14
image_epoch = 14
do_generate_single_image = True

do_generate_single_gif_chrono_order = False
do_generate_single_gif_image_order = True
do_generate_combo_gif = True

# Zmienne globalne
raw_data: mne.io.Raw
epochs: mne.Epochs
tfData_float32: np.float32

#---------------------------------------------Wczytywanie danych--------------------------
def load_data(nick):
    global raw_data

    raw_data = mne.io.read_raw_edf(f"assets/{nick}_raw.edf", preload=True, infer_types=True) # załaduj do pamięci, typ EEG do znanych kanałów
    print("\n#############################################\n")
    print(raw_data) #Informacje o kanałach i objekcie
    print("\n#############################################\n")
    print(raw_data.info) #Informacje o kanałach i objekcie
    print("\n#############################################\n")
    print(raw_data.describe()) #Informacje o kanałach i objekcie
    print("\n#############################################\n")
#---------------------------------------------Zmiana nazw kanałów-------------------------
def change_channel_names():
    global raw_data

    new_names = { ch:ch.split('-')[0].strip() for ch in raw_data.ch_names}
    raw_data.rename_channels(new_names)
#---------------------------------------------Zmiana typów kanałów------------------------
def change_channel_types():
    global raw_data

    ch_types = {}
    for ch in raw_data.ch_names:
        if ch in ['X1:', 'X2:', 'X3:', 'CM']:
            ch_types[ch] = 'misc'
        elif ch in ['Trigger', 'Event']:
            ch_types[ch] = 'stim'
        elif ch[:3] == "Imp":
            raw_data.drop_channels([ch])
        else:
            ch_types[ch] = 'eeg'
    raw_data.set_channel_types(ch_types)
#--------------------------------------------Zmiana montażu i referencji-------------------
def change_montage_and_reference():
    global raw_data

    montage = mne.channels.make_standard_montage('standard_1020')
    raw_data.set_montage(montage, on_missing='ignore')

    raw_data.set_eeg_reference(ref_channels=['A1', 'A2'])
#--------------------------------------------Czyszczenie pustych kanałów-------------------
def drop_unimportant_channels():
    global raw_data
    raw_data.drop_channels(['Event']) #Pominiecie elektrod z ktorych jest 0 sygnalu
#--------------------------------------------Filtrowanie-----------------------------------
def filter_raw_data():
    global raw_data
    raw_data = raw_data.notch_filter(freqs=50, verbose=False)
    raw_data = raw_data.filter(l_freq=1, h_freq=40, verbose=False)
#-------------------------Automatyczne wykrywanie złych kanałów na podstawie impedancji----
def delete_bad_channels():
# 1. Kanały z bardzo dużym udziałem Low_SNR oznaczamy jako "bad"
# 2. Dla pozostałych kanałów tworzymy adnotacje czasowe BAD_*
#    aby odrzucać tylko fragmenty nagrania o złej jakości
    df_imp = pd.read_csv(f"assets/{NICK_BADANEGO}_imp.csv", skiprows=6)

    bad_channels = []
    annotations = []

    excluded_imp_channels = ["X1:", "X2:", "X3:", "CM"]             # Kanały pomocnicze - nie analizujemy ich jakości EEG
    dead_channel_threshold = 0.50                                   # Kanał uznajemy za martwy jeśli >50% próbek ma Low_SNR

    time_step = df_imp["Time"].diff().median()          # Maksymalna przerwa (w sekundach) między kolejnymi wpisami Low_SNR,
                                                        # aby traktować je jako jeden ciągły fragment
    merge_gap = time_step * 1.5                         # merge_gap = 1.5 # bezpieczna, stała wartość,
                                                        #gdy kolumna Time jest właśnie próbkowana co 1 sekundę
    for col in df_imp.columns:
        if col == "Time":
            continue
        if col in excluded_imp_channels:
            print(f"{col}: pominięty (kanał pomocniczy)")
            continue

        mask = ( df_imp[col].astype(str).str.strip() == 'Low_SNR')
        low_snr_sum = mask.sum()
        low_snr_ratio = low_snr_sum / len(df_imp)
        print(f"{col}: {low_snr_ratio:.2%} Low_SNR")

        if low_snr_ratio > dead_channel_threshold:                   # MARTWY KANAŁ - oznaczamy cały kanał jako bad
            bad_channels.append(col)
            print(
                f"  -> kanał oznaczony jako BAD "
                f"(Low_SNR = {low_snr_ratio:.2%})"
            )
            continue

        bad_times = df_imp.loc[mask, "Time"].to_numpy()              # KANAŁ W WIĘKSZOŚCI DOBRY - tworzymy adnotacje czasowe
        if len(bad_times) == 0:
            continue

        start = bad_times[0]                # znacznik początku obecnie rozważanego fragmenu
        prev = bad_times[0]                 # znacznik poprzedniej rozważanej części fragmentu

        for current in bad_times[1:]:
            if current - prev > merge_gap:
                annotations.append(
                    dict(
                        onset=float(start),
                        duration=float(prev - start + 1),
                        description=f"BAD_{col}"
                    )
                )
                start = current
            prev = current
        # ostatni fragment
        annotations.append(
            dict(
                onset=float(start),
                duration=float(prev - start + 1),
                description=f"BAD_{col}"
            )
        )

    bads_in_edf = []                                                         # Mapowanie nazw kanałów z CSV na nazwy w EDF
    for bad_ch in bad_channels:
        matched = [ch for ch in raw_data.ch_names if bad_ch in ch]
        bads_in_edf.extend(matched)

    raw_data.info['bads'] = bads_in_edf
    print(f"Updated MNE Raw object. Bad channels: {raw_data.info['bads']}")


    if len(annotations) > 0:                 # Tworzenie adnotacji MNE
        annot = mne.Annotations(
            onset=[a["onset"] for a in annotations],
            duration=[a["duration"] for a in annotations],
            description=[a["description"] for a in annotations],
            orig_time=raw_data.annotations.orig_time

        )
        raw_data.set_annotations(
            raw_data.annotations + annot
        )
        print(
            f"Dodano {len(annotations)} "
            f"adnotacji związanych z Low_SNR."
        )
#----------------------Analiza Komponentów Sygnału---------------------------------------------
def independent_component_anlysis():
    global raw_data

    ica = mne.preprocessing.ICA(
        n_components=0.99,
        method="fastica",
        random_state=42,
        max_iter="auto"
    )

    print("Dopasowywanie ICA...")
    ica.fit(raw_data)

    eog_inds, scores = ica.find_bads_eog(                 # Wykrywanie komponentów związanych z mrugnięciami (EOG)
        raw_data,
        ch_name=["X1:", "X2:"]
    )
    print(f"Znalezione komponenty EOG: {eog_inds}")

    ica.exclude.extend(eog_inds)                                    # Oznaczenie komponentów do usunięcia
    try:
        muscle_inds, muscle_scores = ica.find_bads_muscle(          # Wykrywanie artefaktów mięśniowych
            raw_data
        )
        print(f"Znalezione komponenty mięśniowe: {muscle_inds}")

        if len(muscle_inds) > 0:
            # UWAGA:
            # Na początku warto obejrzeć komponenty ręcznie
            # przed automatycznym usuwaniem.
            ica.exclude.extend(muscle_inds)
    except Exception as e:
        print("Nie udało się wykryć komponentów mięśniowych:")
        print(e)

    print(f"Usuwane komponenty ICA: {ica.exclude}") # Usunięcie wybranych komponentów
    raw_data = ica.apply(raw_data.copy())
    print("ICA zakończona.")
#----------------------Tworzenie i wizualizacja Eventów----------------------------------------
def create_epochs_from_ImageOn_events():
    global raw_data
    global epochs

    events = mne.find_events(raw_data, stim_channel="Trigger", initial_event=True, shortest_event=0.5, min_duration=0.001, consecutive=True)
    print(f"Liczba wszystkich eventów: {len(events)}")
    #----------------------Filtracja eventu IMAGE_ON---------------------------------------------------
    imageOn_events = events[(events[:, 2] == 12)]
#---------------------Porównanie Ilości znalezionych eventów z dziennikiem w CSV-------------------
    events_csv = pd.read_csv(f"assets/{CSV_EVENTY}")
    image_events_csv = ( events_csv.query("event_code == 12").reset_index(drop=True) ) #przeszukanie csv za IMAGE_ON
    print( f"Liczba IMAGE_ON w CSV: {len(image_events_csv)}")

    assert len(imageOn_events) == len(image_events_csv), \
        "Liczba eventów IMAGE_ON w EEG i w CSV nie jest taka sama!"
#---------------------Utworzenie metadanych dla epok-----------------------------------------------
    metadata = image_events_csv[
        [ "participant_id", "series_id", "trial_id",
            "image_id","image_category","image_file" ]
    ].copy()

    event_id = {"IMAGE_ON": 12}             # Definicja event_id
#---------------------Utworzenie epok-----------------------------------------------
    epochs = mne.Epochs(                 # t = 0      -> pojawienie się obrazka
        raw_data,                        # t = -0.35   -> 350 ms przed obrazkiem
        imageOn_events,                  # t = 0.5    -> 500 ms po rozpoczęciu wyświetlania
        event_id=event_id,
        tmin=-0.35,
        tmax=0.5,
        baseline=(-0.35, -0.01),
        metadata=metadata,
        preload=True,
        reject_by_annotation=True # odrzucenie epok z adnotacjami BAD_
    )
    usuniete = [idx for idx, log in enumerate(epochs.drop_log) if len(log) > 0]
    print(f"Lista usuniętych epok: {(usuniete)}")
#------------------Analiza w Time-Freq używając falek Morleta-----------------------
def morlet_wavelet():
    global epochs
    global tfData_float32

    czestotliwosci = np.arange(4,40, 1)
    l_cykli = czestotliwosci/4
    tfa = epochs.compute_tfr(method="morlet", freqs=czestotliwosci, n_cycles=l_cykli,
                        decim=1, picks='eeg', return_itc=False)

    if reduct_tfa_using_baseline:       #redukcja sygnału względem funkcji na w odniesieniu do sygnału przed wydarzeniem
        tfa.apply_baseline(baseline=(-0.3, -0.05), mode=chosen_reduction_method)

    tfData = tfa.crop(tmin=0.0, tmax=0.5).get_data()
    tfData_float32 = tfData.astype(np.float32)

    if subtract_mean_baseline:
        srednia = np.mean(tfData_float32, axis=1, keepdims=True)			#LEKKIE POLEPSZENIE WYNIKOW
        tfData_float32 = tfData_float32-srednia

    print(tfData_float32)
#-------------------Przygotowanie zmiennych do klasyfikacji--------------------------
def save_to_file(nick):
    y_category = epochs.metadata["image_category"]	# Kategoria obrazka
    y_image = epochs.metadata["image_id"]		# Klasyfikacja według pliku

    print("Category y")
    print(y_category.head())
    print("Image y")
    print(y_image.head())
    #-----------------Zapis wyników do plików-------------------------------------------

    sub_mean_base_text = "AVG"                              #Ustalanie nazwy pliku na podstawie wybranych opcji
    reduct_text = ""
    if subtract_mean_baseline:
        sub_mean_base_text = "NoAVG"
    if reduct_tfa_using_baseline:
        reduct_text = chosen_reduction_method
    #NOCLIP zawsze, clip daje zle efekty

    np.save(f"data/{nick}_Dane32Przetworzone{sub_mean_base_text}{reduct_text}.npy", tfData_float32)
    np.save(f"data/{nick}_EtykietyDanych.npy", y_category)
    print("Utworzono Pliki")
# Pojedynczy obrazek dla jednej epoki dla jednego kanału
def generate_single_epoche_image():
    global tfData_float32
    global image_epoch
    global image_channel

    jedenWynik = tfData_float32[image_epoch, image_channel, :, :]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(jedenWynik, cmap="jet", aspect="auto", origin="lower")

    ax.set_title(f"Epoka {image_epoch}, Kanał {image_channel}")
    ax.set_xlabel("Punkty czasu")
    ax.set_ylabel("Indeksy częstotliwości")
    fig.colorbar(im, label="Moc sygnału")

    plt.savefig(f"{visualization_catalogue}/{NICK_BADANEGO}_spektrogram_epoka_{image_epoch}.png", dpi=100, bbox_inches="tight")
    plt.show()
    plt.close()
    print("Obrazek PNG został zapisany!")
# GIF dla jednego kanału z róznych epok po kolei
def generate_single_channel_gif_in_chrono_order():
    global tfData_float32
    nr_kanalu = 14          # GIF dla jednego kanału z róznych epok po kolei
    frames = []

    for nr_epoki in range(200):
        fig, ax = plt.subplots(figsize=(8, 6))

        single_spectrogram = tfData_float32[nr_epoki, nr_kanalu, :, :]
        ax.imshow(single_spectrogram, cmap="jet", aspect="auto", origin="lower")
        ax.set_title(f"Epoka: {nr_epoki}")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)

        frames.append(iio.imread(buf))
        plt.close(fig)

    # duration=1000 sekudna na klatke
    iio.imwrite(f"{visualization_catalogue}/{NICK_BADANEGO}_animacja_epok_chrono.gif", frames, duration=250, loop=0)
    print("GIF utworzony!")
# GIF dla jednego knału z epok posortowanych według obrazka
def generate_single_channel_gif_in_image_order():
    y_image = epochs.metadata["image_id"]  # Klasyfikacja według pliku
    sort_indek = np.argsort(y_image)

    nr_kanalu = 14
    frames = []

    for nrKlatki, nr_epoki in enumerate(sort_indek[120:1199]):
        fig, ax = plt.subplots(figsize=(8, 6))

        single_spectrogram = tfData_float32[nr_epoki, nr_kanalu, :, :]

        obecnyobraz = y_image[nr_epoki]

        ax.imshow(single_spectrogram, cmap="jet", aspect="auto", origin="lower")
        ax.set_title(f"Kanał: {raw_data.ch_names[nr_kanalu]} "
                     f"Epoka: {nr_epoki} Obraz {obecnyobraz}")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)

        frames.append(iio.imread(buf))
        plt.close(fig)

    # duration=1000 sekudna na klatke
    iio.imwrite(f"{visualization_catalogue}/{NICK_BADANEGO}_animacja_epok.gif", frames, duration=400, loop=0)
    print("GIF utworzony!")
#GIF dla elektrod F3, F4, C3, C4 O1, O2 dla posortoawnych obrazków
def generate_combo_GIF_image_order():
    y_category = epochs.metadata["image_category"]	# Kategoria obrazka
    y_image = epochs.metadata["image_id"]		# Klasyfikacja według pliku

    y_cat_np = y_category.to_numpy()
    y_img_np = y_image.to_numpy()
    sort_indek = np.argsort(y_img_np)

    frames = []
    kanalyL = [2,1,14]
    kanalyP = [4,5,15]

    warunek_zm = y_cat_np[sort_indek[:-1]] != y_cat_np[sort_indek[1:]]
    p_zmiany = np.where(warunek_zm)[0] + 1
    granice = [0] + list(p_zmiany) + [len(sort_indek)]

    for i in range(0,len(granice)-1):
        print(y_cat_np[sort_indek[granice[i]]] )
    obKat = 10
    for nrKlatki,nr_epoki in enumerate(sort_indek[granice[obKat]:granice[obKat+1]]):
        fig, ax = plt.subplots(3, 2, figsize=(8, 6))

        obecnyobraz= y_img_np[nr_epoki]

        for idy, nr_kanalu in enumerate(kanalyL):
            single_spectrogram = tfData_float32[nr_epoki, nr_kanalu, :, :]

            ax[idy,0].imshow(single_spectrogram, cmap="jet", aspect="auto", origin="lower")
            ax[idy,0].set_title(f"Kanał: {raw_data.ch_names[nr_kanalu]} ")

        for idy, nr_kanalu in enumerate(kanalyP):
            single_spectrogram = tfData_float32[nr_epoki, nr_kanalu, :, :]

            ax[idy,1].imshow(single_spectrogram, cmap="jet", aspect="auto", origin="lower")
            ax[idy,1].set_title(f"Kanał: {raw_data.ch_names[nr_kanalu]}")

        fig.suptitle(f"Obraz {obecnyobraz}", fontsize=10, fontweight='bold')
        ax[2,1].set_xlabel("Czas")
        ax[1,0].set_ylabel("Częstotliwość")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)

        frames.append(iio.imread(buf))
        plt.close(fig)

    # duration=1000 sekudna na klatke
    iio.imwrite(f"{visualization_catalogue}/{NICK_BADANEGO}_animacja_epokKanalow_{now}.gif", frames, duration=600, loop=0)
    print("Piękny GIF utworzony!")


for i in wybrani_badani:
    NICK_BADANEGO = nicki_badanych[i]
    CSV_EVENTY = csv_eventow[i]

    load_data(NICK_BADANEGO)
    change_channel_names()
    change_channel_types()
    change_montage_and_reference()
    drop_unimportant_channels()
    filter_raw_data()
    delete_bad_channels()
    independent_component_anlysis()
    create_epochs_from_ImageOn_events()
    morlet_wavelet()
    if do_save_to_file:
        save_to_file(NICK_BADANEGO)
    if do_generate_single_image:
        generate_single_epoche_image()
    if do_generate_single_gif_chrono_order:
        generate_single_channel_gif_in_chrono_order()
    if do_generate_single_gif_image_order:
        generate_single_channel_gif_in_image_order()
    if do_generate_combo_gif:
        generate_combo_GIF_image_order()
