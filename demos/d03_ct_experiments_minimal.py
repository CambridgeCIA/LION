import LION.experiments.ct_experiments as ct_experiments

experiment = ct_experiments.clinicalCTRecon(dataset="LIDC-IDRI")
experiment.param.noise_params.I0 = 2000

training_dataset = experiment.get_training_dataset()
validation_dataset = experiment.get_validation_dataset()
testing_dataset = experiment.get_testing_dataset()

import LION.CTtools.ct_geometry as ctgeo
from LION.data_loaders.LIDC_IDRI import LIDC_IDRI
from LION.experiments.ct_experiments import Experiment
from LION.utils.parameter import LIONParameter


class TestExperiment(Experiment):
    def __init__(self, experiment_params=None, dataset="LIDC-IDRI"):
        super().__init__(experiment_params, dataset)

    # The only other method you need to define is default_parameters(), e.g. this is the one for clinicalCT
    @staticmethod
    def default_parameters(dataset="LIDC-IDRI"):
        param = LIONParameter()
        param.name = "Clinical dose experiment"
        # Parameters for the geometry
        param.geo = ctgeo.Geometry.default_parameters()
        # Parameters for the noise in the sinogram.
        param.noise_params = LIONParameter()
        param.noise_params.I0 = 10000
        param.noise_params.sigma = 5
        param.noise_params.cross_talk = 0.05

        if dataset == "LIDC-IDRI":
            # Parameters for the LIDC-IDRI dataset
            param.data_loader_params = LIDC_IDRI.default_parameters(
                geo=param.geo, task="reconstruction"
            )
            param.data_loader_params.max_num_slices_per_patient = 5
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented")
        return param


# Now you have a class, you can just call it
experiment = TestExperiment()


# TODO: Is it really a good idea to encourage users to always write a new experiment class
#       like lines 91 to 120 above,
#       instead of passing custom parameters in their own script for a new experiment like below?
#       Wouldn't this make people feel like they need to learn how to develop LION,
#       make a subclass, override methods, etc., just to do a simple experiment?
#       (If they make mistakes constructing an object and passing custom parameters,
#       it seems even more likely they would make mistakes creating a new class).
#       Should we encourage users to not do simple throw-away experiments using LION
#       even during exploration phase?

geo_2 = ctgeo.Geometry.default_parameters()
# Can be shorter if `max_num_slices_per_patient` can be passed to the constructor
data_loader_params_2 = LIDC_IDRI.default_parameters(geo=geo_2, task="reconstruction")
data_loader_params_2.max_num_slices_per_patient = 5
experiment_2 = Experiment(  # Assuming Experiment can be initialized directly
    experiment_params=LIONParameter(
        name="Clinical dose experiment",
        geo=geo_2,  # can be shorter if `geo` is inferred from `data_loader_params`
        noise_params=LIONParameter(I0=10000, sigma=5, cross_talk=0.05),
        data_loader_params=data_loader_params_2,
    )
)


# More ideal:
experiment_3 = Experiment(  # Assuming Experiment can be initialized directly
    experiment_params=LIONParameter(  # Assuming `geo` is inferred from `data_loader_params`
        name="Clinical dose experiment",
        noise_params=LIONParameter(I0=10000, sigma=5, cross_talk=0.05),
        data_loader_params=LIDC_IDRI.default_parameters(
            geo_2, "reconstruction", max_num_slices_per_patient=5
        ),
    )
)
