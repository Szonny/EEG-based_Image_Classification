import numpy as np

nicki_badanych = ["abc", "Bear", "fghx", "jt", "mi2", "miguel", "mole", "Reshi", "sapling"]
nazwy_plikow = ["_Dane32PrzetworzoneAVG.npy"]
nazwy_plikow_etykiet = "_EtykietyDanych.npy"
obecny_typ_pliku = nazwy_plikow[0]
poczatki_przedzialow = [0] * (len(nicki_badanych) + 1)

#--------------------Liczenie sumy dlugosci----------------
suma_dlugosci = 0

for it in range(0, len(nicki_badanych)):
    tablica = np.load(f"data/{nicki_badanych[it]}{obecny_typ_pliku}")
    poczatki_przedzialow[it] = suma_dlugosci
    suma_dlugosci += len(tablica)
poczatki_przedzialow[len(nicki_badanych)] = suma_dlugosci
#--------------------Tworzenie nowych indeksow-------------
def f_mieszajaca(maks):
    return np.random.randint(maks)

indeksy = [-1] * suma_dlugosci
for i in range(0, suma_dlugosci):
    nowy_indeks = f_mieszajaca(suma_dlugosci)
    while indeksy[nowy_indeks] != -1:
        nowy_indeks = nowy_indeks + 1
        nowy_indeks = nowy_indeks % suma_dlugosci
    indeksy[nowy_indeks] = i
#--------------------Zapisywanie do plików wynikowych--------
rozwazany_indeks = 0
dlugosc_wynikowych = int((suma_dlugosci)/(len(nicki_badanych)))

for nr_pliku_wynikowego in range(0,len(nicki_badanych)): #dla każdego pliku wynikowego
    wynikowy_tensor = [0] * dlugosc_wynikowych
    wynikowe_etykiety = [""] * dlugosc_wynikowych

    for i in range(0, len(nicki_badanych)): #dla kazdego pliku
        tablica = np.load(f"data/{nicki_badanych[i]}{obecny_typ_pliku}")
        etykiety = np.load(f"data/{nicki_badanych[i]}{nazwy_plikow_etykiet}")

        for j in range(0 , dlugosc_wynikowych):   #znajdz indeksy odpowiadajace plikowi
            if indeksy[j+rozwazany_indeks] >= poczatki_przedzialow[i] and indeksy[j+rozwazany_indeks] < poczatki_przedzialow[i+1]:
                wynikowy_tensor[j] = tablica[indeksy[j+rozwazany_indeks]-poczatki_przedzialow[i]]
                wynikowe_etykiety[j] = etykiety[indeksy[j+rozwazany_indeks]-poczatki_przedzialow[i]]

    rozwazany_indeks += (dlugosc_wynikowych-1)

    np.save(f"data/plikWynikowy{nr_pliku_wynikowego}{nazwy_plikow}",wynikowy_tensor)
    np.save(f"data/Etykiety{nr_pliku_wynikowego}{nazwy_plikow_etykiet}",wynikowe_etykiety)
#--------------------Reszta-danych----------------------------------------
if ((suma_dlugosci)%(len(nicki_badanych)))>0:
    wynikowy_tensor = [0] * ((suma_dlugosci)%(len(nicki_badanych)))
    wynikowe_etykiety = [""] * ((suma_dlugosci)%(len(nicki_badanych)))

    for i in range(0, len(nicki_badanych)):  # dla kazdego pliku
        tablica = np.load(f"data/{nicki_badanych[i]}{obecny_typ_pliku}")
        etykiety = np.load(f"data/{nicki_badanych[i]}{nazwy_plikow_etykiet}")

        for j in range(0, (suma_dlugosci)%(len(nicki_badanych))):  # znajdz indeksy odpowiadajace plikowi
            if indeksy[j + rozwazany_indeks] >= poczatki_przedzialow[i] and indeksy[j + rozwazany_indeks] < \
                    poczatki_przedzialow[i + 1]:
                wynikowy_tensor[j] = tablica[indeksy[j + rozwazany_indeks] - poczatki_przedzialow[i]]
                wynikowe_etykiety[j] = etykiety[indeksy[j + rozwazany_indeks] - poczatki_przedzialow[i]]

    np.save(f"data/plikWynikowy_remains{nazwy_plikow}", wynikowy_tensor)
    np.save(f"data/Etykiety_remains{nazwy_plikow_etykiet}", wynikowe_etykiety)