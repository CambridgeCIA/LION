import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f_oneway
import glob

experiment_type = ["LowDose","SparseView", "LimitedAngle", "ExLowDose"]

inp = -1

while(not (inp in range(4))):
    print("SELECT ONE OF THE FOLLOWING EXPERIMENTS BY TYPING THE NUMBER\n")
    for i, exp_name in enumerate(experiment_type):
        print(f"{i}:   {exp_name}")
    try:
        inp = int(input("0, 1, 2, 3?\n"))
    except:
        inp = -1

experiment_type = experiment_type[inp]

#Just add model names here and csv path and it should work as long as MODELNAME_test.py was written properly
model_names = -1
csv_path = ""
csv_path = f"./../CSV/{experiment_type}/"

csv_files = glob.glob(csv_path + "/*.csv")

for i , path in enumerate(csv_files):
    if (i == 0):
        model_names = []

    model_names.append(path[len(csv_path)::].replace("/","").replace(".csv",""))

if (model_names == -1):
    print("No CSV files found (line 34)")
    exit()
#list of the csv header's names for the metrics
data_types = ['SSIM', 'MSE' ,'PSNR']

dataframes = {}


#Read each MODELNAME_test's output csvs and combines them into 1 pandas dataframe. Does not repeat values for FBP
for i in range(len(model_names)):
    path_name = csv_path + model_names[i] + ".csv"
    if (i == 0):
        df = pd.read_csv(path_name)
        continue

    mf = pd.read_csv(path_name)
    for dtypes in data_types:
        df[model_names[i] + "_" + dtypes] = mf[model_names[i] + "_" + dtypes]
        


#Ask if there are new models to write to results.txt in the CTImages25/CSV
#saves mean median mode and box plot info into a dataframe
#Then saves the string version of that table into a txt file

rewrite = input("Does results.txt need to be updated?\n")
if(rewrite.lower().replace(" ","") in ['yes','y']):
    with open(csv_path+"results.txt", 'w') as f:
            pd.set_option('display.max_columns', None)
            f.write(str(df.describe(include='all')))



#ANOVA

model_names.append("FBP")
model_names.append("SIRT50")
model_names.append("SIRT_LEARNED")
print("Which models would you like to include in this covariance test?")
print("Your options are:")
for m in model_names:
    print(m)
print("All")
chosen_models = input("Please enter the desired models for testing separated by commas\n (case sensetive, spaces don't matter, ex. ADZnet,ItNet)\n")
chosen_models = model_names if (chosen_models.lower() == "all") else chosen_models.replace(" ","").split(",")


print("Which metrics would you like to include in your ANOVA test?")
print("Your options are:")

for m in data_types:
    if(m == "FBP"):
        print("May or may not exist depending on the model:")
    print(m)
print("all")
chosen_metrics = input("Please enter the desired metrics separated by commas (spaces don't matter)\n")
chosen_metrics = data_types if (chosen_metrics.lower() == "all") else chosen_metrics.upper().replace(" ","").split(",")


#prints results of ANOVA

for metric in chosen_metrics:
    ls = []
    # try:
    for model in chosen_models:
        try:
            ls.append(df[model + "_" + metric])
        except:
            print(f"\nCould not find {metric} for {model}")
            pass


    f_stats, p_value = f_oneway(*ls)

    print("\n=======================================")
    print(f"{metric} for {str([mod.name for mod in ls])}")
    print(f"F Statistics: {f_stats: .5f}")
    print(f"P value: {p_value: .5f}\n")
    print("=======================================")
    # except(Exception):
    #     print(Exception)
    #     print(f"\nCould not find {metric} for {str(model)}")
