import numpy as np
import os

from fontTools.varLib.instancer import __main__

FILE_PREFIXES = ["abc", "Bear", "fghx", "jt", "mi2", "miguel", "mole", "Reshi", "sapling"]

#np.save(f"data/{nick}_data32Przetworzone{sub_mean_base_text}{reduct_text}.npy", tfData_float32)
#np.save(f"data/{nick}_labelsDanych.npy{sub_mean_base_text}{reduct_text}", y_category)

reduction_methods = ["", "ratio", "logratio", "zlogratio", "mean", "zscore" , "RAW"]
mean_infixes = ["AVG", "NoAVG"]

file_names = ["_data32Przetworzone"]
file_names_eti = ["_labelsDanych"]
prefixes = ["","TRAIN","TEST","VALIDATE"]

current_file_type = file_names[0]
current_file_type_et = file_names_eti[0]
current_reduction_method = reduction_methods[0]
if_mean_deleted = mean_infixes[0]

do_morlet=True                      # Czy wykonane zostało TFA, Czy są to surowe data??

if do_morlet:
    current_file_type = file_names[0]
    current_file_type_et = file_names_eti[0]
else:
    current_file_type = file_names[1]
    current_file_type_et = file_names_eti[1]

def shuffle_data(patients_nicknames, result_postfix):
    postfix = f"{current_file_type}{if_mean_deleted}{current_reduction_method}.npy"
    postfixET = f"{current_file_type_et}{if_mean_deleted}{current_reduction_method}.npy"
    
    sections_beginnings = [0] * (len(patients_nicknames) + 1)
    #--------------------Liczenie sumy dlugosci----------------
    length_sum = 0
    channel_value_sum = np.zeros(21, dtype=np.float32)
    channel_mean = np.zeros(21, dtype=np.float32)
    sum_elements_amount_interchannel = 0 #suma elementów we wszystkich oobrazkach (kanały-warstwy)

    for it in range(0, len(patients_nicknames)):
        array = np.load(f"data/{patients_nicknames[it]}{postfix}", allow_pickle=True)
        sections_beginnings[it] = length_sum
        length_sum += len(array)
        channel_value_sum += np.sum(array, axis=(0,2,3))
        sum_elements_amount_interchannel += array.shape[0]*array.shape[2]*array.shape[3]
        sections_beginnings[len(patients_nicknames)] = length_sum
    if sum_elements_amount_interchannel > 0:
        channel_mean = channel_value_sum / sum_elements_amount_interchannel
    #--------------------Tworzenie nowych indexow-------------
    indicies = np.arange(0, length_sum)
    np.random.shuffle(indicies)
    print("Utworzono indicies, zaczynam rodzielac data")
    #--------------------Zapisywanie do plików tymczasowych--------
    results_length = int((length_sum)/(len(patients_nicknames)))
    results_amount = len(patients_nicknames)

    for source_index in range(0,len(patients_nicknames)):              #dla każdego zrodla
        source = np.load(f"data/{patients_nicknames[source_index]}{postfix}", allow_pickle=True)
        labels = np.load(f"data/{patients_nicknames[source_index]}{postfixET}", allow_pickle=True)

        for result_no in range(0, results_amount):              #dla kazdego pliku wynikowego
            data = []
            label_pack = []
            beg_file_index = results_length*result_no
            end_file_index = results_length*(result_no+1)

            for i in range(beg_file_index , end_file_index):         #dla kazdego elementu w wynikowym
                if_i_in_range = sections_beginnings[source_index] <= indicies[i] < sections_beginnings[source_index + 1]

                if if_i_in_range:      #jesli element jest z obecnie wczytanego zrodla
                    indices_in_file = indicies[i] - sections_beginnings[source_index]
                    data.append(source[indices_in_file])                                   #to go zapisz
                    label_pack.append(labels[indices_in_file])
            if len(data) > 0:
                np.save(f"data2/tymczasowy_z{source_index}-w{result_no}.npy", data)
                np.save(f"data2/tymczasowyET_z{source_index}-w{result_no}.npy", label_pack)

        if length_sum % (len(patients_nicknames)) > 0 and results_length*(results_amount)< length_sum:    #Resztki
            data = []
            label_pack = []
            for i in range(results_length*(results_amount) , length_sum):
                if_i_in_range = sections_beginnings[source_index] <= indicies[i] < sections_beginnings[source_index + 1]
                if if_i_in_range:
                    data.append(source[indicies[i]- sections_beginnings[source_index]])
                    label_pack.append(labels[indicies[i]- sections_beginnings[source_index]])
            if len(data) > 0:
                np.save(f"data2/tymczasowy_z{source_index}-r.npy", data)
                np.save(f"data2/tymczasowyET_z{source_index}-r.npy", label_pack)
    #--------------------Zapisywanie do plików wynikowych--------
    print("data poszatkowane, lacze data na nowo w nowej kolejnosci")

    for result_no in range(0, results_amount):
        data = []
        et=[]
        for fragment in range(0, len(patients_nicknames)):
            p_src = f"data2/tymczasowy_z{fragment}-w{result_no}.npy"
            p_et = f"data2/tymczasowyET_z{fragment}-w{result_no}.npy"
            source = np.load(f"data2/tymczasowy_z{fragment}-w{result_no}.npy")
            labels = np.load(f"data2/tymczasowyET_z{fragment}-w{result_no}.npy")
            data.append(source)
            et.append(labels)
            os.remove(p_src)
            os.remove(p_et)
        data = np.concatenate(data, axis=0)
        et = np.concatenate(et, axis = 0)

        miniindicies = np.arange(0,len(data))
        np.random.shuffle(miniindicies)

        data = data[miniindicies]
        et = et[miniindicies]

        np.save(f"data2/plikWynikowy{result_no}_{result_postfix}{postfix}", data)
        np.save(f"data2/plikWynikowyET{result_no}_{result_postfix}{postfix}", et)

    if length_sum % (len(patients_nicknames)) > 0:
        data = []
        et=[]
        for result_no in range(0, results_amount):
            if os.path.exists(f"data2/tymczasowy_z{result_no}-r.npy"):
                p_src= f"data2/tymczasowy_z{result_no}-r.npy"
                p_et = f"data2/tymczasowyET_z{result_no}-r.npy"
                source =    np.load(p_src, allow_pickle=True)
                labels =  np.load(p_et, allow_pickle=True)
                data.append(source)
                et.append(labels)
                os.remove(p_src)
                os.remove(p_et)
        data = np.concatenate(data, axis = 0)
        et = np.concatenate(et, axis = 0)
        np.save(f"data2/plikWynikowy{results_amount}_{result_postfix}{postfix}",data)
        np.save(f"data2/plikWynikowyET{results_amount}_{result_postfix}{postfix}", et)

        #--------------------Sprawdzenie-----------------------------------------------
    print("Sprawdzam poprawnosc danych")

    data = np.load(f"data/{patients_nicknames[0]}{postfix}", allow_pickle=True)
    labels = np.load(f"data/{patients_nicknames[0]}{postfixET}", allow_pickle=True)

    first100 = data[0:100]
    labels100 = labels[0:100]
    found = [False] * 100

    sum_of_diff_squared = np.zeros(21, dtype=np.float32)

    for result_no in range(0, results_amount):
        data = np.load(f"data2/plikWynikowy{result_no}_{result_postfix}{postfix}")
        labels = np.load(f"data2/plikWynikowyET{result_no}_{result_postfix}{postfix}")

        mean_format = channel_mean.reshape(1,-1,1,1) #Średnie dla kanałów
        sum_of_diff_squared += np.sum((data - mean_format) ** 2,  axis=(0,2,3))

        for index in range(0,results_length):
            for elZ100 in range(0,100):
                if np.array_equal(data[index], first100[elZ100]) and np.array_equal(labels100[elZ100], labels[index]):
                    found[elZ100] = True

    #REEEEESZZTTTYYYY
    if (os.path.exists(f"data2/plikWynikowy{results_amount}_{result_postfix}{postfix}") and
            length_sum % (len(patients_nicknames)) > 0 and results_length*(results_amount)< length_sum):
        data = np.load(f"data2/plikWynikowy{results_amount}_{result_postfix}{postfix}")
        labels = np.load(f"data2/plikWynikowyET{results_amount}_{result_postfix}{postfix}")
        
        remains_size = length_sum-(results_amount*results_length)

        mean_format = channel_mean.reshape(1,-1,1,1) #Średnie dla kanałów
        sum_of_diff_squared += np.sum((data - mean_format) ** 2, axis=(0,2,3))

        for index in range(0,remains_size):
            for elZ100 in range(0,100):
                if np.array_equal(data[index], first100[elZ100]) and np.array_equal(labels100[elZ100], labels[index]):
                    found[elZ100] = True

    if sum(found) == 100:
        print(f"Udalo sie utworzyc pliki. Znaleziono grupe testowa w wynikowych! Plik data2/plikWynikowy_{result_postfix}{postfix} poprawny. Przystepuje do normalizacji danych")
    else:
        print(f"BLAD. Niezgodnosc etykiet z grupy testowej lub brak niektorych elementow! Znaleziono zgodnych: {sum(found)}")


    std_deviation = np.sqrt(sum_of_diff_squared/sum_elements_amount_interchannel)
    parameters = []

    mean_format = channel_mean.reshape(1, -1, 1, 1)
    std_format = std_deviation.reshape(1, -1, 1, 1)

    parameters.append(mean_format)
    parameters.append(std_format)
    np.save(f"data2/parameters_{result_postfix}{if_mean_deleted}{current_reduction_method}.npy", parameters)
    #-------------------Normalizacja (jeli zażadano TFA)------------------------------------------------
    file_amount = results_amount
    if length_sum % (len(patients_nicknames)) > 0:
        file_amount += 1
    for result_no in range(0, file_amount):
        data = np.load(f"data2/plikWynikowy{result_no}_{result_postfix}{postfix}")
        
        if result_postfix == "TRAIN":
            print("Paczka Treningowa, normalizuje...")
            data = (data-mean_format)/std_format
        else:
            print("Paczka Testowa, normalizuje parametrami z danych treningowych.")
            if os.path.exists(f"data2/parameters_TRAIN{if_mean_deleted}{current_reduction_method}.npy"):
                parameters=np.load(f"data2/parameters_TRAIN{if_mean_deleted}{current_reduction_method}.npy")
                mean_format = parameters[0]
                std_format = parameters[1]
        data = (data-mean_format)/std_format
        
        data = np.transpose(data, (0, 2, 3, 1))
        np.save(f"data2/plikWynikowy{result_no}_{result_postfix}{postfix}",data)
        print(f"data znormalizowane i zapisane w data2/plikWynikowy{result_no}_{result_postfix}{postfix}")

def generate_TRAIN():
    result_postfix = prefixes[1]
    patients_nicknames = FILE_PREFIXES[0:7]
    #patients_nicknames = [FILE_PREFIXES[0]]
    shuffle_data(patients_nicknames, result_postfix)

def generate_VALIDATE():
    result_postfix = prefixes[3]
    patients_nicknames = FILE_PREFIXES[5:7]
    shuffle_data(patients_nicknames, result_postfix)

def generate_TEST():
    result_postfix = prefixes[2]
    patients_nicknames = FILE_PREFIXES[7:]
    shuffle_data(patients_nicknames, result_postfix)

def generate_result(do_TFA = True, if_baseline_red=False, reduction_method = 2, us_sre=False):
    global current_file_type, current_file_type_et, current_reduction_method, if_mean_deleted

    if not os.path.exists("data2"):
        os.mkdir("data2")

    current_file_type = file_names[0]
    current_file_type_et = file_names_eti[0]
    current_reduction_method = reduction_methods[reduction_method+1]
    if not do_TFA:
        current_reduction_method = reduction_methods[6]
    if not if_baseline_red:
        current_reduction_method = reduction_methods[0]
    if_mean_deleted = mean_infixes[us_sre]

    global do_morlet
    do_morlet = do_TFA
    generate_TRAIN()
    generate_TEST()

if __name__ == "__main__":
    generate_result(True)