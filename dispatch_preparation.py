import data_preparation
import data_randomizer

import subprocess

from data_preparation import generuj_pliki
from data_randomizer import generuj_wynikowe

do_tfa_morlet = True
jedenstrzal = False

if jedenstrzal:
    print("###########################################")
    print("Generuje pliki")
    print("###########################################")

    #subprocess.run(["python", "data_preparation.py"])
    generuj_pliki([0, 1, 2, 3, 4, 5, 6, 7, 8],do_tfa_morlet, morlet_log=True, subtract_mean=False, reduct_tfa_baseline = False, red_method=2)

    print("###########################################")
    print("Wygenerowano pliki, mieszam dane!")
    print("###########################################")

    #subprocess.run(["python", "data_randomizer.py"])
    generuj_wynikowe(do_tfa_morlet, czy_redukcja_basln=False, metoda_red = 2, us_sre=False)

    print("###########################################")
    print("Pomieszalem dane!")
    print("###########################################")
else:           #Hurtowe generowanie danych >100 GB!!
    generuj_pliki([0, 1, 2, 3, 4, 5, 6, 7, 8], do_tfa_morlet, morlet_log=True, subtract_mean=False,
                  reduct_tfa_baseline=False, red_method=2)
    generuj_wynikowe(do_tfa_morlet, czy_redukcja_basln=False, metoda_red = 2, us_sre=False)

    generuj_pliki([0, 1, 2, 3, 4, 5, 6, 7, 8], do_tfa_morlet, morlet_log=True, subtract_mean=True,
                  reduct_tfa_baseline=False, red_method=2)
    generuj_wynikowe(do_tfa_morlet, czy_redukcja_basln=False, metoda_red = 2, us_sre=True)

    generuj_pliki([0, 1, 2, 3, 4, 5, 6, 7, 8], do_tfa_morlet, morlet_log=True, subtract_mean=False,
                  reduct_tfa_baseline=True, red_method=0)       #"ratio"
    generuj_wynikowe(do_tfa_morlet, czy_redukcja_basln=True, metoda_red = 0, us_sre=False)

    generuj_pliki([0, 1, 2, 3, 4, 5, 6, 7, 8], do_tfa_morlet, morlet_log=True, subtract_mean=False,
                  reduct_tfa_baseline=True, red_method=1)           #"logratio"
    generuj_wynikowe(do_tfa_morlet, czy_redukcja_basln=True, metoda_red = 1, us_sre=False)

    generuj_pliki([0, 1, 2, 3, 4, 5, 6, 7, 8], do_tfa_morlet, morlet_log=True, subtract_mean=False,
                  reduct_tfa_baseline=True, red_method=2)       #"zlogratio"
    generuj_wynikowe(do_tfa_morlet, czy_redukcja_basln=True, metoda_red = 2, us_sre=False)

    generuj_pliki([0, 1, 2, 3, 4, 5, 6, 7, 8], do_tfa_morlet, morlet_log=True, subtract_mean=False,
                  reduct_tfa_baseline=True, red_method=3)           #"mean"
    generuj_wynikowe(do_tfa_morlet, czy_redukcja_basln=True, metoda_red = 3, us_sre=False)

    generuj_pliki([0, 1, 2, 3, 4, 5, 6, 7, 8], do_tfa_morlet, morlet_log=True, subtract_mean=False,
                  reduct_tfa_baseline=True, red_method=4)       #"zscore"
    generuj_wynikowe(do_tfa_morlet, czy_redukcja_basln=True, metoda_red = 4, us_sre=False)