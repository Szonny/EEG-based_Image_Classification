import numpy as np

nicki_badanych = ["abc", "Bear", "fghx", "jt", "mi2", "miguel", "mole", "Reshi", "sapling"]
nazwy_plikow = ["_Dane32PrzetworzoneAVG.npy"]
nazwy_plikow_etykiet = "_EtykietyDanych.npy"
obecny_typ_pliku = nazwy_plikow[0]
poczatki_przedzialow = [0] * (len(nicki_badanych) + 1)

#--------------------Liczenie sumy dlugosci----------------
suma_dlugosci = 0

for it in range(0, len(nicki_badanych)):
    tablica = np.load(f"data/{nicki_badanych[it]}{obecny_typ_pliku}", allow_pickle=True)
    poczatki_przedzialow[it] = suma_dlugosci
    suma_dlugosci += len(tablica)
poczatki_przedzialow[len(nicki_badanych)] = suma_dlugosci
#--------------------Tworzenie nowych indeksow-------------
indeksy = np.arange(0, suma_dlugosci)
np.random.shuffle(indeksy)
#--------------------Zapisywanie do plików tymczasowych--------
rozwazany_indeks = 0
dlugosc_wynikowych = int((suma_dlugosci)/(len(nicki_badanych)))
ilosc_wynikowych = len(nicki_badanych)

for nr_zrodla in range(0,len(nicki_badanych)):              #dla każdego zrodla
    zrodlo = np.load(f"data/{nicki_badanych[nr_zrodla]}{obecny_typ_pliku}", allow_pickle=True)
    etykiety = np.load(f"data/{nicki_badanych[nr_zrodla]}{nazwy_plikow_etykiet}", allow_pickle=True)

    for nr_wynikowego in range(0, ilosc_wynikowych):              #dla kazdego pliku wynikowego
        dane = []
        paczka_etykiet = []

        for i in range(dlugosc_wynikowych*nr_wynikowego , dlugosc_wynikowych*(nr_wynikowego+1)):         #dla kazdego elementu w wynikowym
            if indeksy[i] >= poczatki_przedzialow[nr_zrodla] and indeksy[i]<poczatki_przedzialow[nr_zrodla+1]:      #jesli element jest z obecnie wczytanego zrodla
                dane.append(zrodlo[indeksy[i] - poczatki_przedzialow[nr_zrodla]])                                   #to go zapisz
                paczka_etykiet.append(etykiety[indeksy[i] - poczatki_przedzialow[nr_zrodla]])
        if len(dane) > 0:
            np.save(f"data2/tymczasowy_z{nr_zrodla}-w{nr_wynikowego}.npy", dane)
            np.save(f"data2/tymczasowyET_z{nr_zrodla}-w{nr_wynikowego}.npy", paczka_etykiet)

    if suma_dlugosci % (len(nicki_badanych)) > 0 and dlugosc_wynikowych*(ilosc_wynikowych)< suma_dlugosci:    #Resztki
        dane = []
        paczka_etykiet = []
        for i in range(dlugosc_wynikowych*(ilosc_wynikowych) , suma_dlugosci):
            if indeksy[i] >= poczatki_przedzialow[nr_zrodla] and indeksy[i]<poczatki_przedzialow[nr_zrodla+1]:
                dane.append(zrodlo[indeksy[i]- poczatki_przedzialow[nr_zrodla]])
                paczka_etykiet.append(etykiety[indeksy[i]- poczatki_przedzialow[nr_zrodla]])
        if len(dane) > 0:
            np.save(f"data2/tymczasowy_z{nr_zrodla}-r.npy", dane)
            np.save(f"data2/tymczasowyET_z{nr_zrodla}-r.npy", paczka_etykiet)
#--------------------Zapisywanie do plików wynikowych--------

for nr_wynikowego in range(0, ilosc_wynikowych):
    dane = []
    et=[]
    for fragment in range(0, len(nicki_badanych)):
        zrodlo = np.load(f"data2/tymczasowy_z{fragment}-w{nr_wynikowego}.npy")
        etykiety = np.load(f"data2/tymczasowyET_z{fragment}-w{nr_wynikowego}.npy")
        dane.append(zrodlo)
        et.append(etykiety)
    dane = np.concatenate(dane, axis=0)
    et = np.concatenate(et, axis = 0)

    miniindeksy = np.arange(0,len(dane))
    np.random.shuffle(miniindeksy)

    dane = dane[miniindeksy]
    et = et[miniindeksy]

    np.save(f"data2/plikWynikowy{nr_wynikowego}_Dane32PrzetworzoneAVG.npy", dane)
    np.save(f"data2/plikWynikowyET{nr_wynikowego}_Dane32PrzetworzoneAVG.npy", et)

if suma_dlugosci % (len(nicki_badanych)) > 0:
    dane = []
    et=[]
    for nr_wynikowego in range(0, ilosc_wynikowych):
        zrodlo = np.load(f"data2/tymczasowy_z{nr_wynikowego}-r.npy")
        etykiety = np.load(f"data2/tymczasowyET_z{nr_wynikowego}-r.npy")
        dane.append(zrodlo)
        et.append(etykiety)
    dane = np.concatenate(dane, axis = 0)
    et = np.concatenate(et, axis = 0)
    np.save(f"data2/plikWynikowy{ilosc_wynikowych}_Dane32PrzetworzoneAVG.npy",dane)
    np.save(f"data2/plikWynikowyET{ilosc_wynikowych}_Dane32PrzetworzoneAVG.npy", et)


