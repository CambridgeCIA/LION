import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pathlib
import pdb
import LION.CTtools.ct_utils as ct
import glob
#change for diff model
from LION.models.iterative_unrolled.LPD import LPD
print("LPD imported")
from LION.models.iterative_unrolled.LPD_SIRT import LPD_SIRT
print("LPD_SIRT imported")
from LION.models.iterative_unrolled.ItNet import ItNet
print("ItNet imported")
from LION.models.iterative_unrolled.ItNet_TV import ItNet as ItNetTV
print("ItNetTV imported")
from LION.models.iterative_unrolled.LG import LG
print("LG imported")
from LION.models.iterative_unrolled.ADZnet import ItNetSobel
print("ItNetSobel imported")
from LION.models.iterative_unrolled.LEARN import LEARN
print("LEARN imported")
# from LION.models.iterative_unrolled.

string_to_class = {}
string_to_class['LPD_FBP'] = LPD
string_to_class['LPD_SIRT'] = LPD_SIRT
string_to_class['ItNet_FBP'] = ItNet
string_to_class['ItNetTV_FBP'] = ItNetTV
string_to_class['LG_FBP'] = LG
string_to_class['ItNetSobel_FBP'] = ItNetSobel
string_to_class['LEARN_FBP'] = LEARN

EXP = ["LowDose","SparseView", "LimitedAngle", "ExLowDose"]

# final_result_fname["LowDose_LPD"] = pathlib.Path("/gscratch/uwb/CTImages25/Scripts/ALL_MODELS/LPD.pt")


models_path = f"./ALL_MODELS/"

model_files = glob.glob(models_path + "*.pt")

counted = {}

#[MODELNAMECLASS(STRING), EXPERIMENT(STRING), MODIFIER(STRING), pt FILE PATH]
model_paths = []
for i , path in enumerate(model_files):
    file_name = path[len(models_path)::].replace("/","")

    fart = file_name.replace(".pt","").split("_")
    fart.append(file_name)
    counted[fart[0]] = False
    model_paths.append(fart)

from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments

datafolder = pathlib.Path(
    "/gscratch/uwb/LION_data/processed/LIDC-IDRI/"
)
#change path for diff model
def experiment_load_buffer(exptype):
    if exptype == "LowDose":
        return ct_experiments.LowDoseCTRecon(datafolder = datafolder)
    elif exptype == "SparseView":
        return ct_experiments.SparseAngleCTRecon(datafolder = datafolder)
    elif exptype == "LimitedAngle":
        return ct_experiments.LimitedAngleCTRecon(datafolder = datafolder)
    elif exptype == "ExLowDose":
        return ct_experiments.ExtremeLowDoseCTRecon(datafolder = datafolder)
    else:
        print(f"{exptype}")
        print("UNRECOGNIZED EXPERIMENT TYPE LINE 137")
        quit()
#[MODELNAMECLASS(STRING), EXPERIMENT(STRING), MODIFIER(STRING), pt FILE PATH]





#CHANGE LOADING
for curr_model in model_paths:
    # experiment = experiment_load_buffer(curr_model[1])
    # op = ct.make_operator(experiment.geometry)
    # dataset = experiment.get_testing_dataset()
    # batch_size = 1
    # dataloader = DataLoader(dataset, batch_size, shuffle=False)
    try:
        MODEL_model, MODEL_param, MODEL_data = string_to_class[f"{curr_model[0]}_{curr_model[2]}"].load("/gscratch/uwb/CTImages25/Scripts/ALL_MODELS/"+curr_model[3])
    except:
        print(str(curr_model))
        continue
    if(counted[curr_model[0]]):
        continue

    counted[curr_model[0]] = True

    def count_parameters(model):
        total = 0
        for param in model.parameters():
            num_params = param.numel()
            total += num_params
        return total


    paramet=count_parameters(MODEL_model)
    print(f"{curr_model[0]}_{curr_model[2]} has {paramet:,} parameters.")

