import subprocess


print("###########################################")
print("Generuje pliki")
print("###########################################")

subprocess.run(["python", "data_preparation.py"])

print("###########################################")
print("Wygenerowano pliki, mieszam dane!")
print("###########################################")

subprocess.run(["python", "data_randomizer.py"])

print("###########################################")
print("Pomieszalem dane!")
print("###########################################")