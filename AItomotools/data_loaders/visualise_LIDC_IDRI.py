from AItomotools.data_loaders.LIDC_IDRI import LIDC_IDRI
import matplotlib.pyplot as plt
import numpy as np
import pylidc as pl
import json
import collections
from AItomotools.data_loaders.LIDC_IDRI import is_nodule_on_slice
from tqdm import tqdm

class Visualise_LIDC_IDRI:
    """
    Class to visualise data of LIDC-IDRI dataset

    Implemented methods:
    * plot_each_patient_with_their_number_of_slices: x_axis -> patient id, y_axis -> total number of slices of this patient
    * plot_histogram: x_axis -> number of slices, y_axis -> total number of patients with this amount of slices
    * plot_each_patient_with_their_number_of_nodules: x_axis -> patient id, y_axis -> total number of nodules of this patient
    * plot_slices_per_patient_that_contain_at_least_one_nodule: x_axis -> patient id, y_axis -> total number of slices that contain at least one nodule of this patient
    * plot_slices_per_patient_that_contain_more_than_one_nodule: x_axis -> patient id, y_axis -> total number of slices that contain more than one nodule of this patient
    * plot_slice_thickness_per_patient: x_axis -> patient id, y_axis -> total slice_thickness of this patient
    
    """

    def __init__(self, savepath):

        self.savepath = savepath
        self.data = LIDC_IDRI("segmentation", 1, "training")
        self.patient_id_with_number_of_slices = self.data.patient_index_to_n_slices_dict

    
    def plot_each_patient_with_their_number_of_slices(self, save_bar_plot_as_png=True, save_values_in_txt=False):
        """
        Creates a plot for each patient in LIDC IDRI dataset with their total number of slices
        """

        # New dict to store patient index and number of slices
        tempdict = {}

        # Change keys in dict from 'LIDC-IDRI-0001' to '1' for plot visualisation
        for index, key in enumerate(self.patient_id_with_number_of_slices):
            tempdict[index+1] = self.patient_id_with_number_of_slices[key]

        if save_values_in_txt:
            # Write key value pairs in .txt file
            with open(self.savepath + "patient_with_number_of_slices.txt", 'w') as f: 
                for key, value in tempdict.items(): 
                    f.write('%s %s\n' % (key, value))

        if save_bar_plot_as_png:
            # Create bar plot for total number of slices per patient
            fig = plt.figure(figsize=(16,6))
            plt.bar(range(len(tempdict)), list(tempdict.values()))
            plt.axhline(np.mean(list(tempdict.values())), color='r')
            ax = plt.gca()
            ax.set_xlim([0, 1012])
            ax.set_ylim([0, 800])
            plt.title('Total number of slices per patient in LIDC-IDRI dataset.')
            plt.xlabel('Patient id')
            plt.ylabel('Total number of slices')
            plt.annotate(f'Mean: {np.mean(list(tempdict.values())):.0f}', xy = (0.75, 0.85), xycoords='figure fraction')
            plt.annotate(f'Min: {np.min(list(tempdict.values())):.0f}', xy = (0.75, 0.80), xycoords='figure fraction')
            plt.annotate(f'Max: {np.max(list(tempdict.values())):.0f}', xy = (0.75, 0.75), xycoords='figure fraction')
            plt.savefig(self.savepath + 'patient_with_number_of_slices.png')
            plt.close()

    def plot_histogram(self, save_values_in_txt=False):
        """
        Creates a histogram plot LIDC-IDRI dataset. (= Total number of patients with n slices)
        """

        fig = plt.figure(figsize=(16,6))
        bins = np.arange(0,800,1)
        y, x, _ = plt.hist(self.patient_id_with_number_of_slices.values(), bins=bins)
        ax = plt.gca()
        ax.set_xlim([0, 800])
        ax.set_ylim([0, 75])
        plt.title('Histogram: Total number of patients with x slices.')
        plt.xlabel('Total number of slices')
        plt.ylabel('Total number of patients')
        plt.savefig(self.savepath + 'histogram.png')
        plt.close()

        if save_values_in_txt:
            # Write key value pairs in .txt file
            with open(self.savepath + "histogram.txt", 'w') as f: 
                for index in range(len(y)):
                    f.write('%s %s\n' % (int(x[index]), int(y[index])))

    def plot_each_patient_with_their_number_of_nodules(self, save_bar_plot_as_png=True, save_values_in_txt=False):
        """
        Creates a plot for each patient in LIDC-IDRI dataset with their number of nodules
        """

        # Load dict from json file
        patient_index_to_n_nodules_dict = json.load(open("/local/scratch/public/AItomotools/processed/LIDC-IDRI/patient_id_to_n_nodules.json"))

        # Adding missing values
        patient_index_to_n_nodules_dict['LIDC-IDRI-0238'] = 0
        patient_index_to_n_nodules_dict['LIDC-IDRI-0585'] = 0

        # Reorder dict because of added values that are missing in json file
        patient_index_to_n_nodules_dict = collections.OrderedDict(sorted(patient_index_to_n_nodules_dict.items()))

        tempdict={}

        # Change keys in dict from 'LIDC-IDRI-0001' to '1' for plot visualisation
        for index, key in enumerate(patient_index_to_n_nodules_dict):
            tempdict[index+1] = patient_index_to_n_nodules_dict[key]

        if save_values_in_txt:
            # Write key value pairs in .txt file
            with open(self.savepath + "patient_with_number_of_nodules.txt", 'w') as f: 
                for key, value in tempdict.items(): 
                    f.write('%s %s\n' % (key, value))

        if save_bar_plot_as_png:
            # Create bar flot for total number of nodules per patient
            fig = plt.figure(figsize=(16,6))
            plt.bar(range(len(tempdict)), list(tempdict.values()))
            plt.axhline(np.mean(list(tempdict.values())), color='r')
            ax = plt.gca()
            ax.set_xlim([0, 1012])
            ax.set_ylim([0, 30])
            plt.title('Total number of nodules per patient in LIDC-IDRI dataset')
            plt.xlabel('Patient id')
            plt.ylabel('Total number of nodules')
            plt.annotate(f'Mean: {np.mean(list(tempdict.values())):.0f}', xy = (0.75, 0.85), xycoords='figure fraction')
            plt.annotate(f'Min: {np.min(list(tempdict.values())):.0f}', xy = (0.75, 0.80), xycoords='figure fraction')
            plt.annotate(f'Max: {np.max(list(tempdict.values())):.0f}', xy = (0.75, 0.75), xycoords='figure fraction')
            plt.savefig(self.savepath + 'patient_with_number_of_nodules.png')
            plt.close()

    def plot_slices_per_patient_that_contain_at_least_one_nodule(self, save_values_in_txt=False):
        """
        Creates a plot for each patient in LIDC-IDRI dataset with their number of slices that contain at least one nodule
        WARNING: Currently really slow computation! Work on how to speedup is in progress... 
        """

        print("Preparing slices of each patient contain at least one nodule, this may take time....")

        slices_with_nodules_for_patient = {}
        for patientid in tqdm(self.patient_id_with_number_of_slices): # Loop over all patients
            nodule_slice_counter = 0
            for slice in range(self.patient_id_with_number_of_slices[patientid]): # Loop over all slices of each patient
                scan: pl.Scan = (pl.query(pl.Scan).filter(pl.Scan.patient_id == patientid).first()) # Get information about this patient
                list_of_annotated_nodules = scan.cluster_annotations(verbose=False) # List of annotated nodules of this patient
                breaker = False
                for nodule in range(len(list_of_annotated_nodules)): # Loop over all nodules of this patient
                    for nodule_annotation in list_of_annotated_nodules[nodule]: # Loop over all annotations of this nodule
                        if is_nodule_on_slice(nodule_annotation, slice)[0]:
                            nodule_slice_counter+=1
                            breaker = True
                            break
                    if breaker: break
            slices_with_nodules_for_patient[patientid] = nodule_slice_counter

        # New dict to store patient index and number of slices
        tempdict = {}

        # Change keys in dict from 'LIDC-IDRI-0001' to '1' for plot visualisation
        for index, key in enumerate(self.patient_id_with_number_of_slices):
            tempdict[index+1] = self.patient_id_with_number_of_slices[key]

        # Change keys in dict from 'LIDC-IDRI-0001' to '1' for plot visualisation
        patient_nodule_dict={}
        for index, key in enumerate(slices_with_nodules_for_patient):
            patient_nodule_dict[index+1] = slices_with_nodules_for_patient[key]

        if save_values_in_txt: 
            # Write key value pairs in .txt file
            with open(self.savepath + "number_of_slices_per_patient_that_contain_at_least_one_nodule.txt", 'w') as f: 
                for key, value in tempdict.items(): 
                    f.write('%s %s\n' % (key, value))
        
        # Create bar flot for total number of slices per patient that contain at least one nodule
        fig = plt.figure(figsize=(16,6))
        plt.bar(range(len(tempdict)), list(tempdict.values()), color='blue')
        plt.bar(range(len(patient_nodule_dict)), list(patient_nodule_dict.values()), color ='green')
        plt.axhline(np.mean(list(patient_nodule_dict.values())), color='r')
        plt.axhline(np.mean(list(patient_nodule_dict.values())), color='r')
        ax = plt.gca()
        ax.set_xlim([0, 1012])
        ax.set_ylim([0, 800])
        plt.title('Total number of slices per patient in LIDC-IDRI dataset')
        plt.xlabel('Patient id')
        plt.ylabel('Total number of slices')
        plt.annotate(f'Mean: {np.mean(list(tempdict.values())):.0f}', xy = (0.75, 0.85), xycoords='figure fraction')
        plt.annotate(f'Min: {np.min(list(tempdict.values())):.0f}', xy = (0.75, 0.80), xycoords='figure fraction')
        plt.annotate(f'Max: {np.max(list(tempdict.values())):.0f}', xy = (0.75, 0.75), xycoords='figure fraction')
        plt.savefig(self.savepath + 'number_of_slices_per_patient_that_contain_at_least_one_nodule_with_total_amount_of_slices.png')
        plt.close()


        fig = plt.figure(figsize=(16,6))
        plt.bar(range(len(patient_nodule_dict)), list(patient_nodule_dict.values()))
        plt.axhline(np.mean(list(patient_nodule_dict.values())), color='r')
        ax = plt.gca()
        ax.set_xlim([0, 1012])
        ax.set_ylim([0, 150])
        plt.title('Total number of slices that contain at least one nodule per patient in LIDC-IDRI dataset')
        plt.xlabel('Patient id')
        plt.ylabel('Total number of slices that contain at least one nodule')
        plt.annotate(f'Mean: {np.mean(list(patient_nodule_dict.values())):.0f}', xy = (0.75, 0.85), xycoords='figure fraction')
        plt.annotate(f'Min: {np.min(list(patient_nodule_dict.values())):.0f}', xy = (0.75, 0.80), xycoords='figure fraction')
        plt.annotate(f'Max: {np.max(list(patient_nodule_dict.values())):.0f}', xy = (0.75, 0.75), xycoords='figure fraction')
        plt.savefig(self.savepath + 'number_of_slices_per_patient_that_contain_at_least_one_nodule.png')
        plt.close()

    def plot_slices_per_patient_that_contain_more_than_one_nodule(self, save_values_in_txt=False):
        """
        Creates a plot for each patient in LIDC-IDRI dataset with their number of slices that contain more than one nodule
        WARNING: Currently really slow computation! Work on how to speedup is in progress... 
        """
        
        print("Preparing slices of each patient contain more than one nodule, this may take time....")
        
        slices_with_nodules_for_patient = {}
        for patientid in tqdm(self.patient_id_with_number_of_slices): # Loop over all patients
            nodule_slice_counter = 0
            for slice in range(self.patient_id_with_number_of_slices[patientid]): # Loop over all slices of each patient
                scan: pl.Scan = (pl.query(pl.Scan).filter(pl.Scan.patient_id == patientid).first()) # Get information about this patient
                list_of_annotated_nodules = scan.cluster_annotations(verbose=False) # List of annotated nodules of this patient
                breaker = False
                one_nodule_found = False
                for nodule in range(len(list_of_annotated_nodules)): # Loop over all nodules of this patient
                    for nodule_annotation in list_of_annotated_nodules[nodule]: # Loop over all annotations of this nodule
                        if is_nodule_on_slice(nodule_annotation, slice)[0] and not one_nodule_found:
                            one_nodule_found = True
                            break
                        if is_nodule_on_slice(nodule_annotation, slice)[0] and one_nodule_found:
                            nodule_slice_counter += 1
                            breaker = True
                            break
                    if breaker: break
            slices_with_nodules_for_patient[patientid] = nodule_slice_counter
        
        # New dict to store patient index and number of slices
        tempdict = {}

        # Change keys in dict from 'LIDC-IDRI-0001' to '1' for plot visualisation
        for index, key in enumerate(self.patient_id_with_number_of_slices):
            tempdict[index+1] = self.patient_id_with_number_of_slices[key]

        # Change keys in dict from 'LIDC-IDRI-0001' to '1' for plot visualisation
        patient_nodule_dict={}
        for index, key in enumerate(slices_with_nodules_for_patient):
            patient_nodule_dict[index+1] = slices_with_nodules_for_patient[key]

        if save_values_in_txt:
            # Write key value pairs in .txt file
            with open(self.savepath + "number_of_slices_per_patient_that_contain_more_than_one_nodule.txt", 'w') as f: 
                for key, value in patient_nodule_dict.items(): 
                    f.write('%s %s\n' % (key, value))

        # Create bar flot for total number of slices per patient that contain more than one nodule
        fig = plt.figure(figsize=(16,6))
        plt.bar(range(len(tempdict)), list(tempdict.values()), color='blue')
        plt.bar(range(len(patient_nodule_dict)), list(patient_nodule_dict.values()), color ='green')
        plt.axhline(np.mean(list(tempdict.values())), color='r')
        plt.axhline(np.mean(list(patient_nodule_dict.values())), color='r')
        ax = plt.gca()
        ax.set_xlim([0, 1012])
        ax.set_ylim([0, 800])
        plt.title('Total number of slices per patient in LIDC-IDRI dataset')
        plt.xlabel('Patient id')
        plt.ylabel('Total number of slices')
        plt.annotate(f'Mean: {np.mean(list(tempdict.values())):.0f}', xy = (0.75, 0.85), xycoords='figure fraction')
        plt.annotate(f'Min: {np.min(list(tempdict.values())):.0f}', xy = (0.75, 0.80), xycoords='figure fraction')
        plt.annotate(f'Max: {np.max(list(tempdict.values())):.0f}', xy = (0.75, 0.75), xycoords='figure fraction')
        plt.savefig(self.savepath + 'number_of_slices_per_patient_that_contain_more_than_one_nodule_with_total_amount_of_slices.png')
        plt.close()


        fig = plt.figure(figsize=(16,6))
        plt.bar(range(len(patient_nodule_dict)), list(patient_nodule_dict.values()))
        plt.axhline(np.mean(list(patient_nodule_dict.values())), color='r')
        ax = plt.gca()
        ax.set_xlim([0, 1012])
        ax.set_ylim([0, 100])
        plt.title('Total number of slices that contain more than one nodule per patient in LIDC-IDRI dataset')
        plt.xlabel('Patient id')
        plt.ylabel('Total number of slices that contain more than one nodule')
        plt.annotate(f'Mean: {np.mean(list(patient_nodule_dict.values())):.0f}', xy = (0.75, 0.85), xycoords='figure fraction')
        plt.annotate(f'Min: {np.min(list(patient_nodule_dict.values())):.0f}', xy = (0.75, 0.80), xycoords='figure fraction')
        plt.annotate(f'Max: {np.max(list(patient_nodule_dict.values())):.0f}', xy = (0.75, 0.75), xycoords='figure fraction')
        plt.savefig(self.savepath + 'number_of_slices_per_patient_that_contain_more_than_one_nodule.png')
        plt.close()

    def plot_slice_thickness_per_patient(self, save_bar_plot_as_png=True, save_values_in_txt=False):
        """
        Creates a plot for each patient in LIDC-IDRI dataset with their total slice thickness
        """
        slice_thickness_for_patient = {}

        for patientid in tqdm(self.patient_id_with_number_of_slices): # Loop over all patients
            scan: pl.Scan = (pl.query(pl.Scan).filter(pl.Scan.patient_id == patientid).first()) # Get information about this patient
            
            # Error handling for patients without any slices. In that case scan is NoneType object and slice thickness for this patient is 0
            if not scan == None:
                slice_thickness_for_patient[patientid] = scan.slice_thickness * self.patient_id_with_number_of_slices[patientid] # Multiply slice thickness with number of slices per patient
            else: 
                slice_thickness_for_patient[patientid] = 0
        # New dict to store patient index and number of slices
        tempdict = {}

        # Change keys in dict from 'LIDC-IDRI-0001' to '1' for plot visualisation
        for index, key in enumerate(slice_thickness_for_patient):
            tempdict[index+1] = slice_thickness_for_patient[key]

        if save_values_in_txt:
            # Write key value pairs in .txt file
            with open(self.savepath + "slice_thickness_per_patient.txt", 'w') as f: 
                for key, value in tempdict.items(): 
                    f.write('%s %s\n' % (key, value))
        
        if save_bar_plot_as_png:
            # Create bar plot
            fig = plt.figure(figsize=(16,6))
            plt.bar(range(len(tempdict)), list(tempdict.values()))
            plt.axhline(np.mean(list(tempdict.values())), color='r')
            ax = plt.gca()
            ax.set_xlim([0, 1012])
            ax.set_ylim([0, 1300])
            plt.title('Total slice thickness per patient in LIDC-IDRI dataset.')
            plt.xlabel('Patient id')
            plt.ylabel('Slice thickness')
            plt.annotate(f'Mean: {np.mean(list(tempdict.values())):.0f}', xy = (0.75, 0.85), xycoords='figure fraction')
            plt.annotate(f'Min: {np.min(list(tempdict.values())):.0f}', xy = (0.75, 0.80), xycoords='figure fraction')
            plt.annotate(f'Max: {np.max(list(tempdict.values())):.0f}', xy = (0.75, 0.75), xycoords='figure fraction')
            plt.savefig(self.savepath + 'slice_thickness_per_patient.png')
            plt.close()

    def create_all_plots(self):
        print('Plot each patient with their number of slices...')
        self.plot_each_patient_with_their_number_of_slices(save_values_in_txt=True)
        print('Plot histogram...')
        self.plot_histogram(save_values_in_txt=True)
        print('Plot each patient with their number of nodules...')
        self.plot_each_patient_with_their_number_of_nodules(save_values_in_txt=True)
        print('Plot slices per patient that contain at least one nodule...')
        self.plot_slices_per_patient_that_contain_at_least_one_nodule(save_values_in_txt=True)
        print('Plot slices per patient that contain more than one nodule...')
        self.plot_slices_per_patient_that_contain_more_than_one_nodule(save_values_in_txt=True)
        print('Plot slice thickness per patient...')
        self.plot_slice_thickness_per_patient(save_values_in_txt=True)

if __name__ == '__main__':
    #  
    # Change savepath
    #
    savepath = '/store/DAMTP/ml2119/'
    Visualise_LIDC_IDRI(savepath).create_all_plots()