import data_preparation
import data_randomizer

import subprocess

from data_preparation import generate_files
from data_randomizer import generate_result

do_tfa_morlet = True
one_shot_generate = True

if one_shot_generate:
    print("###########################################")
    print("Generuje pliki")
    print("###########################################")

    #subprocess.run(["python", "data_preparation.py"])
    generate_files([0, 1, 2, 3, 4, 5, 6, 7, 8],do_tfa_morlet, morlet_log=True, subtract_mean=False, reduct_tfa_baseline = False, red_method=2)

    print("###########################################")
    print("Wygenerowano pliki, mieszam dane!")
    print("###########################################")

    #subprocess.run(["python", "data_randomizer.py"])
    generate_result(do_tfa_morlet, if_baseline_red=False, reduction_method = 2, mean_del=False)

    print("###########################################")
    print("Pomieszalem dane!")
    print("###########################################")
else:           #Hurtowe generowanie danych >100 GB!!
    generate_files([0, 1, 2, 3, 4, 5, 6, 7, 8], do_tfa_morlet, morlet_log=True, subtract_mean=False,
                  reduct_tfa_baseline=False, red_method=2)
    generate_result(do_tfa_morlet, if_baseline_red=False, reduction_method = 2, mean_del=False)

    generate_files([0, 1, 2, 3, 4, 5, 6, 7, 8], do_tfa_morlet, morlet_log=True, subtract_mean=True,
                  reduct_tfa_baseline=False, red_method=2)
    generate_result(do_tfa_morlet, if_baseline_red=False, reduction_method = 2, mean_del=True)

    generate_files([0, 1, 2, 3, 4, 5, 6, 7, 8], do_tfa_morlet, morlet_log=True, subtract_mean=False,
                  reduct_tfa_baseline=True, red_method=0)       #"ratio"
    generate_result(do_tfa_morlet, if_baseline_red=True, reduction_method = 0, mean_del=False)

    generate_files([0, 1, 2, 3, 4, 5, 6, 7, 8], do_tfa_morlet, morlet_log=True, subtract_mean=False,
                  reduct_tfa_baseline=True, red_method=1)           #"logratio"
    generate_result(do_tfa_morlet, if_baseline_red=True, reduction_method = 1, mean_del=False)

    generate_files([0, 1, 2, 3, 4, 5, 6, 7, 8], do_tfa_morlet, morlet_log=True, subtract_mean=False,
                  reduct_tfa_baseline=True, red_method=2)       #"zlogratio"
    generate_result(do_tfa_morlet, if_baseline_red=True, reduction_method = 2, mean_del=False)

    generate_files([0, 1, 2, 3, 4, 5, 6, 7, 8], do_tfa_morlet, morlet_log=True, subtract_mean=False,
                  reduct_tfa_baseline=True, red_method=3)           #"mean"
    generate_result(do_tfa_morlet, if_baseline_red=True, reduction_method = 3, mean_del=False)

    generate_files([0, 1, 2, 3, 4, 5, 6, 7, 8], do_tfa_morlet, morlet_log=True, subtract_mean=False,
                  reduct_tfa_baseline=True, red_method=4)       #"zscore"
    generate_result(do_tfa_morlet, if_baseline_red=True, reduction_method = 4, mean_del=False)