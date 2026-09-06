import numpy as np
import os

from fontTools.varLib.instancer import __main__

PRZYROSTKI_PLIKOW = ["abc", "Bear", "fghx", "jt", "mi2", "miguel", "mole", "Reshi", "sapling"]

#np.save(f"data/{nick}_Dane32Przetworzone{sub_mean_base_text}{reduct_text}.npy", tfData_float32)
#np.save(f"data/{nick}_EtykietyDanych.npy{sub_mean_base_text}{reduct_text}", y_category)

reduction_methods = ["", "ratio", "logratio", "zlogratio", "mean", "zscore" , "RAW"]
wrostki_sredniej = ["AVG", "NoAVG"]

nazwy_plikow = ["_Dane32Przetworzone"]
nazwy_plikow_etykiet = ["_EtykietyDanych"]
przyrostki = ["","TRAIN","TEST","VALIDATE"]

obecny_typ_pliku = nazwy_plikow[0]
obecny_typ_pliku_et = nazwy_plikow_etykiet[0]
obecna_metoda_red = reduction_methods[0]
czy_us_srednia = wrostki_sredniej[0]

do_morlet=True                      # Czy wykonane zostało TFA, Czy są to surowe dane??

if do_morlet:
    obecny_typ_pliku = nazwy_plikow[0]
    obecny_typ_pliku_et = nazwy_plikow_etykiet[0]
else:
    obecny_typ_pliku = nazwy_plikow[1]
    obecny_typ_pliku_et = nazwy_plikow_etykiet[1]

def przemieszaj_dane(nicki_badanych, przyrostek_wynikowy):
    zarostek = f"{obecny_typ_pliku}{czy_us_srednia}{obecna_metoda_red}.npy"
    zarostekET = f"{obecny_typ_pliku_et}{czy_us_srednia}{obecna_metoda_red}.npy"
    
    poczatki_przedzialow = [0] * (len(nicki_badanych) + 1)
    #--------------------Liczenie sumy dlugosci----------------
    suma_dlugosci = 0
    suma_z_kanalow = np.zeros(21, dtype=np.float32)
    srednia_kanalowa = np.zeros(21, dtype=np.float32)
    suma_il_el_miedzykan = 0 #suma elementów we wszystkich oobrazkach (kanały-warstwy)

    for it in range(0, len(nicki_badanych)):
        tablica = np.load(f"data/{nicki_badanych[it]}{zarostek}", allow_pickle=True)
        poczatki_przedzialow[it] = suma_dlugosci
        suma_dlugosci += len(tablica)
        suma_z_kanalow += np.sum(tablica, axis=(0,2,3))
        suma_il_el_miedzykan += tablica.shape[0]*tablica.shape[2]*tablica.shape[3]
        poczatki_przedzialow[len(nicki_badanych)] = suma_dlugosci
    if suma_il_el_miedzykan > 0:
        srednia_kanalowa = suma_z_kanalow / suma_il_el_miedzykan
    #--------------------Tworzenie nowych indeksow-------------
    indeksy = np.arange(0, suma_dlugosci)
    np.random.shuffle(indeksy)
    print("Utworzono indeksy, zaczynam rodzielac dane")
    #--------------------Zapisywanie do plików tymczasowych--------
    rozwazany_indeks = 0
    dlugosc_wynikowych = int((suma_dlugosci)/(len(nicki_badanych)))
    ilosc_wynikowych = len(nicki_badanych)

    for nr_zrodla in range(0,len(nicki_badanych)):              #dla każdego zrodla
        zrodlo = np.load(f"data/{nicki_badanych[nr_zrodla]}{zarostek}", allow_pickle=True)
        etykiety = np.load(f"data/{nicki_badanych[nr_zrodla]}{zarostekET}", allow_pickle=True)

        for nr_wynikowego in range(0, ilosc_wynikowych):              #dla kazdego pliku wynikowego
            dane = []
            paczka_etykiet = []
            pocz_indeks_pliku = dlugosc_wynikowych*nr_wynikowego
            kon_indeks_pliku = dlugosc_wynikowych*(nr_wynikowego+1)

            for i in range(pocz_indeks_pliku , kon_indeks_pliku):         #dla kazdego elementu w wynikowym
                czyWZakresieI = poczatki_przedzialow[nr_zrodla] <= indeksy[i] < poczatki_przedzialow[nr_zrodla + 1]

                if czyWZakresieI:      #jesli element jest z obecnie wczytanego zrodla
                    indeks_w_pliku = indeksy[i] - poczatki_przedzialow[nr_zrodla]
                    dane.append(zrodlo[indeks_w_pliku])                                   #to go zapisz
                    paczka_etykiet.append(etykiety[indeks_w_pliku])
            if len(dane) > 0:
                np.save(f"data2/tymczasowy_z{nr_zrodla}-w{nr_wynikowego}.npy", dane)
                np.save(f"data2/tymczasowyET_z{nr_zrodla}-w{nr_wynikowego}.npy", paczka_etykiet)

        if suma_dlugosci % (len(nicki_badanych)) > 0 and dlugosc_wynikowych*(ilosc_wynikowych)< suma_dlugosci:    #Resztki
            dane = []
            paczka_etykiet = []
            for i in range(dlugosc_wynikowych*(ilosc_wynikowych) , suma_dlugosci):
                czyWZakresieI = poczatki_przedzialow[nr_zrodla] <= indeksy[i] < poczatki_przedzialow[nr_zrodla + 1]
                if czyWZakresieI:
                    dane.append(zrodlo[indeksy[i]- poczatki_przedzialow[nr_zrodla]])
                    paczka_etykiet.append(etykiety[indeksy[i]- poczatki_przedzialow[nr_zrodla]])
            if len(dane) > 0:
                np.save(f"data2/tymczasowy_z{nr_zrodla}-r.npy", dane)
                np.save(f"data2/tymczasowyET_z{nr_zrodla}-r.npy", paczka_etykiet)
    #--------------------Zapisywanie do plików wynikowych--------
    print("Dane poszatkowane, lacze dane na nowo w nowej kolejnosci")

    for nr_wynikowego in range(0, ilosc_wynikowych):
        dane = []
        et=[]
        for fragment in range(0, len(nicki_badanych)):
            p_zrd = f"data2/tymczasowy_z{fragment}-w{nr_wynikowego}.npy"
            p_et = f"data2/tymczasowyET_z{fragment}-w{nr_wynikowego}.npy"
            zrodlo = np.load(f"data2/tymczasowy_z{fragment}-w{nr_wynikowego}.npy")
            etykiety = np.load(f"data2/tymczasowyET_z{fragment}-w{nr_wynikowego}.npy")
            dane.append(zrodlo)
            et.append(etykiety)
            os.remove(p_zrd)
            os.remove(p_et)
        dane = np.concatenate(dane, axis=0)
        et = np.concatenate(et, axis = 0)

        miniindeksy = np.arange(0,len(dane))
        np.random.shuffle(miniindeksy)

        dane = dane[miniindeksy]
        et = et[miniindeksy]

        np.save(f"data2/plikWynikowy{nr_wynikowego}_{przyrostek_wynikowy}{zarostek}", dane)
        np.save(f"data2/plikWynikowyET{nr_wynikowego}_{przyrostek_wynikowy}{zarostek}", et)

    if suma_dlugosci % (len(nicki_badanych)) > 0:
        dane = []
        et=[]
        for nr_wynikowego in range(0, ilosc_wynikowych):
            if os.path.exists(f"data2/tymczasowy_z{nr_wynikowego}-r.npy"):
                p_zrd= f"data2/tymczasowy_z{nr_wynikowego}-r.npy"
                p_et = f"data2/tymczasowyET_z{nr_wynikowego}-r.npy"
                zrodlo =    np.load(p_zrd, allow_pickle=True)
                etykiety =  np.load(p_et, allow_pickle=True)
                dane.append(zrodlo)
                et.append(etykiety)
                os.remove(p_zrd)
                os.remove(p_et)
        dane = np.concatenate(dane, axis = 0)
        et = np.concatenate(et, axis = 0)
        np.save(f"data2/plikWynikowy{ilosc_wynikowych}_{przyrostek_wynikowy}{zarostek}",dane)
        np.save(f"data2/plikWynikowyET{ilosc_wynikowych}_{przyrostek_wynikowy}{zarostek}", et)

        #--------------------Sprawdzenie-----------------------------------------------
    print("Sprawdzam poprawnosc danych")

    dane = np.load(f"data/{nicki_badanych[0]}{zarostek}", allow_pickle=True)
    etykiety = np.load(f"data/{nicki_badanych[0]}{zarostekET}", allow_pickle=True)

    pierwsze100 = dane[0:100]
    etykiety100 = etykiety[0:100]
    znalezione = [False] * 100

    suma_kwadratow_roznic = np.zeros(21, dtype=np.float32)

    for nr_wynikowego in range(0, ilosc_wynikowych):
        dane = np.load(f"data2/plikWynikowy{nr_wynikowego}_{przyrostek_wynikowy}{zarostek}")
        etykiety = np.load(f"data2/plikWynikowyET{nr_wynikowego}_{przyrostek_wynikowy}{zarostek}")

        sr_format = srednia_kanalowa.reshape(1,-1,1,1) #Średnie dla kanałów
        suma_kwadratow_roznic += np.sum((dane - sr_format) ** 2,  axis=(0,2,3))

        for indeks in range(0,dlugosc_wynikowych):
            for elZ100 in range(0,100):
                if np.array_equal(dane[indeks], pierwsze100[elZ100]) and np.array_equal(etykiety100[elZ100], etykiety[indeks]):
                    znalezione[elZ100] = True

    #REEEEESZZTTTYYYY
    if (os.path.exists(f"data2/plikWynikowy{ilosc_wynikowych}_{przyrostek_wynikowy}{zarostek}") and
            suma_dlugosci % (len(nicki_badanych)) > 0 and dlugosc_wynikowych*(ilosc_wynikowych)< suma_dlugosci):
        dane = np.load(f"data2/plikWynikowy{ilosc_wynikowych}_{przyrostek_wynikowy}{zarostek}")
        etykiety = np.load(f"data2/plikWynikowyET{ilosc_wynikowych}_{przyrostek_wynikowy}{zarostek}")
        
        rozmiar_reszty = suma_dlugosci-(ilosc_wynikowych*dlugosc_wynikowych)

        sr_format = srednia_kanalowa.reshape(1,-1,1,1) #Średnie dla kanałów
        suma_kwadratow_roznic += np.sum((dane - sr_format) ** 2, axis=(0,2,3))

        for indeks in range(0,rozmiar_reszty):
            for elZ100 in range(0,100):
                if np.array_equal(dane[indeks], pierwsze100[elZ100]) and np.array_equal(etykiety100[elZ100], etykiety[indeks]):
                    znalezione[elZ100] = True

    if sum(znalezione) == 100:
        print(f"Udalo sie utworzyc pliki. Znaleziono grupe testowa w wynikowych! Plik data2/plikWynikowy_{przyrostek_wynikowy}{zarostek} poprawny. Przystepuje do normalizacji danych")
    else:
        print(f"BLAD. Niezgodnosc etykiet z grupy testowej lub brak niektorych elementow! Znaleziono zgodnych: {sum(znalezione)}")


    odchylenie_std = np.sqrt(suma_kwadratow_roznic/suma_il_el_miedzykan)
    parametry = []

    sr_format = srednia_kanalowa.reshape(1, -1, 1, 1)
    std_format = odchylenie_std.reshape(1, -1, 1, 1)

    parametry.append(sr_format)
    parametry.append(std_format)
    np.save(f"data2/parametry_{przyrostek_wynikowy}{czy_us_srednia}{obecna_metoda_red}.npy", parametry)
    #-------------------Normalizacja (jeli zażadano TFA)------------------------------------------------
    il_plikow = ilosc_wynikowych
    if suma_dlugosci % (len(nicki_badanych)) > 0:
        il_plikow += 1
    for nr_wynikowego in range(0, il_plikow):
        dane = np.load(f"data2/plikWynikowy{nr_wynikowego}_{przyrostek_wynikowy}{zarostek}")
        
        if przyrostek_wynikowy == "TRAIN":
            print("Paczka Treningowa, normalizuje...")
            dane = (dane-sr_format)/std_format
        else:
            print("Paczka Testowa, normalizuje parametrami z danych treningowych.")
            if os.path.exists(f"data2/parametry_TRAIN{czy_us_srednia}{obecna_metoda_red}.npy"):
                parametry=np.load(f"data2/parametry_TRAIN{czy_us_srednia}{obecna_metoda_red}.npy")
                sr_format = parametry[0]
                std_format = parametry[1]
        dane = (dane-sr_format)/std_format
        
        dane = np.transpose(dane, (0, 2, 3, 1))
        np.save(f"data2/plikWynikowy{nr_wynikowego}_{przyrostek_wynikowy}{zarostek}",dane)
        print(f"Dane znormalizowane i zapisane w data2/plikWynikowy{nr_wynikowego}_{przyrostek_wynikowy}{zarostek}")

def generuj_TRAIN():
    przyrostek_wynikowy = przyrostki[1]
    nicki_badanych = PRZYROSTKI_PLIKOW[0:7]
    #nicki_badanych = [PRZYROSTKI_PLIKOW[0]]
    przemieszaj_dane(nicki_badanych, przyrostek_wynikowy)

def generuj_VALIDATE():
    przyrostek_wynikowy = przyrostki[3]
    nicki_badanych = PRZYROSTKI_PLIKOW[5:7]
    przemieszaj_dane(nicki_badanych, przyrostek_wynikowy)

def generuj_TEST():
    przyrostek_wynikowy = przyrostki[2]
    nicki_badanych = PRZYROSTKI_PLIKOW[7:]
    przemieszaj_dane(nicki_badanych, przyrostek_wynikowy)

def generuj_wynikowe(do_TFA = True, czy_redukcja_basln=False, metoda_red = 2, us_sre=False):
    global obecny_typ_pliku, obecny_typ_pliku_et, obecna_metoda_red, czy_us_srednia

    if not os.path.exists("data2"):
        os.mkdir("data2")

    obecny_typ_pliku = nazwy_plikow[0]
    obecny_typ_pliku_et = nazwy_plikow_etykiet[0]
    obecna_metoda_red = reduction_methods[metoda_red+1]
    if not do_TFA:
        obecna_metoda_red = reduction_methods[6]
    if not czy_redukcja_basln:
        obecna_metoda_red = reduction_methods[0]
    czy_us_srednia = wrostki_sredniej[us_sre]

    global do_morlet
    do_morlet = do_TFA
    generuj_TRAIN()
    generuj_TEST()

if __name__ == "__main__":
    generuj_wynikowe(True)