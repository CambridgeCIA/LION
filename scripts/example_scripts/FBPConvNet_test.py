#% This example shows the FBPConvNet Model being tested on unseen data

import pathlib
from LION.models.post_processing.FBPConvNet import FBPConvNet
import LION.experiments.ct_experiments as ct_experiments
from LION.utils.test_model import test_with_img_output

experiment = ct_experiments.clinicalCTRecon()

savefolder = pathlib.Path("/store/DAMTP/cs2186/trained_models/clinical_dose/")
final_result_fname = savefolder.joinpath("FBPConvNet_final_iter.pt")

test_with_img_output(experiment, FBPConvNet, str(final_result_fname), "fbpimg.png", n=3, device="cuda:0")

