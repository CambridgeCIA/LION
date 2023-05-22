import os
import csv
import pathlib

import numpy as np
import skimage.transform
from skimage import filters
from skimage import measure
import skimage.io as skio
from scipy import ndimage


from AItomotools.data_loaders.data_primitives import CTimage
from AItomotools.data_loaders.data_loader import CT_data_loader
import AItomotools.CTtools.ct_utils as ct

## Class that holds LUNA nodule information
## TODO make this more generic, not just LUNA
class Nodule():
    """
    Represents a lung nodule.

    ...

    Attributes
    ----------
    filename : str
        filename of the original file where the nodule is from
    loc : np.array
        location of the nodule in the image
    diameter: float
        diameter of the nodule

    """
    def __init__(self,string) -> None:
        if not isinstance(string,str):
            raise ValueError("Input is not string")
        parts=string.rsplit(sep=',')
        self.filename=parts[0]
        self.loc=np.array([float(i) for i in parts[1:4]])[::-1]
        self.diameter=float(parts[4])
    def __str__(self):
        return f"Nodule \n   file: {self.filename} \n   Coords: [{self.loc[0]}, {self.loc[1]}, {self.loc[2]}] \n  Diameter: {self.diameter}"

#%% Class that contains a LUNA image
# a LUNA image is a CT image + LUNA nodule info
class LunaImage(CTimage):
    def __init__(self,folder:pathlib.Path, file):
        super().__init__(folder, file)
        self.nodule_mask= None
        self.nodules=[]
        self.unit="HU"

    #Genuine question, why do have super() here? Is it related to class inheritance?
    def load_data(self):
        """
        Overloaded data loader, only to be able to do unit coversion if self.unit has changed,
        as CTimage does not know  what is the expected/desired unit
        """
        super().load_data()
        # As we know that LUNA is in HUs, lets check if the units has been changed by the user
        if self.unit=="normal":
            self.data=ct.from_HU_to_normal(self.data)
        if self.unit=="mu":
            self.data=ct.from_HU_to_mu(self.data)

    def unload_data(self):
        self.nodule_mask=None
        super().unload_data()

    def set_nodule_data(self,nodule_list):
        """
        Sets data about Nodules present in this image
        """
        if isinstance(nodule_list,Nodule):
            nodule_list=[nodule_list]
        if not isinstance(nodule_list,list) and not all(isinstance(n, Nodule) for n in nodule_list):
            raise ValueError("input should be Nodule or list of Nodules")
        self.nodules.extend(nodule_list)
        self.nodules=list(set(self.nodules))

    # TODO actually crop the data that is loaded. This is tricky because we need to change the nodule location etc too.
    def get_croped_nodule_slices(self,indices):
        """
        removes all slices from self.data that contain no tumour, given a nodule index
        It does not free the memory, it only returns the cropped data.

        Returns
        ------
        cropped : np.array
            cropped image with only slices that contain tumours. Its only z-copping, individual slices are left untouched
        mask : np.array
            cropped mask with only slices that contain tumours. Its only z-copping, individual slices are left untouched

        """
        if self.nodule_mask is None:
            raise ValueError("Nodule masks required to crop slices")
        if not isinstance(indices,list):
            if indices=="all":
                nodule_list=self.nodules
            elif int(indices) == indices:
                indices=[indices]
                nodule_list=self.nodules[indices]
            else:
                raise ValueError("Input indices need to be an index (or list of indices) of the Nodules")


        cropped=[]
        mask=[]
        for nodule in nodule_list:
            nodule_center=np.round(self.coords2index(nodule.loc)).astype(int)
            nodule_radious=np.ceil(nodule.diameter/self.spacing/2).astype(int)+1
            cropped.append(self.data[nodule_center[0]-nodule_radious[0]:nodule_center[0]+nodule_radious[0]])
            if self.nodule_mask is not None:
                mask.append(self.nodule_mask[nodule_center[0]-nodule_radious[0]:nodule_center[0]+nodule_radious[0]])
        return cropped, mask

    def get_nodule(self,nodule):
        """
        returns a cropped nodule, given we know its radious
        """
        nodule_center=np.round(self.coords2index(nodule.loc)).astype(int)
        nodule_radious=np.ceil(nodule.diameter/self.spacing/2).astype(int)+1
        return self.data[nodule_center[0]-nodule_radious[0]:nodule_center[0]+nodule_radious[0],\
                          nodule_center[1]-nodule_radious[1]:nodule_center[1]+nodule_radious[1],\
                          nodule_center[2]-nodule_radious[2]:nodule_center[2]+nodule_radious[2]]

    def make_nodule_only_mask(self,indices):
        """
        Makes a mask where all the image, except the area where the tumour is, is zeroed.
        """
        if not isinstance(indices,list):
            if indices=="all":
                nodule_list=self.nodules
            elif int(indices) == indices:
                indices=[indices]
                nodule_list=self.nodules[indices]
            else:
                raise ValueError("Input indices need to be an index (or list of indices) of the Nodules")

        if self.nodule_mask is None:
            self.nodule_mask=np.zeros_like(self.data)
        label=max(np.unique(self.nodule_mask))+1
        for nodule in nodule_list:
            nodule_center=np.round(self.coords2index(nodule.loc)).astype(int)
            nodule_radious=np.ceil(nodule.diameter/self.spacing/2).astype(int)+1
            self.nodule_mask[nodule_center[0]-nodule_radious[0]:nodule_center[0]+nodule_radious[0],\
                             nodule_center[1]-nodule_radious[1]:nodule_center[1]+nodule_radious[1],\
                             nodule_center[2]-nodule_radious[2]:nodule_center[2]+nodule_radious[2]]=label
            label=label+1

    def make_binary_mask_nodule(self,indices):
        """
        # DO NOT USE, NOT RELIABLE, OTSU MAY FAIL
        Given a Nodule indices, segment the nodule, make a binary mask.
        This uses Otsu and morphological operations, and its quite flimsy at doing its job.
        Needs work .
        """
        if not isinstance(indices,list):
            if indices=="all":
                nodule_list=self.nodules
            elif int(indices) == indices:
                indices=[indices]
                nodule_list=self.nodules[indices]
            else:
                raise ValueError("Input indices need to be an index (or list of indices) of the Nodules")

        if self.nodule_mask is None:
            self.nodule_mask=np.zeros_like(self.data)*np.nan
        label=max(np.unique(self.nodule_mask))+1
        for nodule in nodule_list:
            nodule_data=self.get_nodule(nodule)
            val=filters.threshold_otsu(nodule_data)
            mask=nodule_data>val
            mask=ndimage.binary_fill_holes(mask)
            nodule_center=np.round(self.coords2index(nodule.loc)).astype(int)
            nodule_radious=np.ceil(nodule.diameter/self.spacing/2).astype(int)+1

            # There will be more blobs than the real stuff in mask, so filter them out.
            blobs_labels = measure.label(mask.astype(float), background=0)
            our_blob=blobs_labels[blobs_labels.shape[0]//2,blobs_labels.shape[1]//2,blobs_labels.shape[2]//2]
            mask=blobs_labels==our_blob

            # now we can put it in the global mask
            self.nodule_mask[nodule_center[0]-nodule_radious[0]:nodule_center[0]+nodule_radious[0],\
                             nodule_center[1]-nodule_radious[1]:nodule_center[1]+nodule_radious[1],\
                             nodule_center[2]-nodule_radious[2]:nodule_center[2]+nodule_radious[2]] = label*mask.astype(float)

            label=label+1



class LUNA16(CT_data_loader):
    """
    Contains the information of the LUNA16 dataset and allows for data loading/processing
    ...

    Attributes
    ----------
    """

    def __init__(self, folder:pathlib.Path, verbose=True,load_metadata=False) -> None:

        self.images=[]
        self.subfolders=None
        self.verbose=verbose
        self.find_subfolders(folder)
        self.unit="HU" # Default in HUs
        if load_metadata:
            self.load_metadata()

    def find_subfolders(self, folder:pathlib.Path):
        """
        Searches LUNA16 dataset for folders with data
        """
        if self.subfolders is not None:
            raise AttributeError("Subfolders already loaded, updating not implemented")
        # This code allows to load partial subfolders, but for now its overriden and loads all.
        subfolder="all"
        self.folder = folder
        if subfolder == "all":
            print(self.folder)
            subfolder = [f.name for f in self.folder.iterdir() if f.is_dir()]
            self.subfolders = [s for s in subfolder if s.startswith("subset")]
        elif '*' in subfolder:
            subfolder_list = [f for f in self.folder.iterdir() if f.is_dir()]
            # Not sure what that does
            self.subfolders = (fnmatch.filter(subfolder_list, subfolder))
        else:
            self.subfolders = [subfolder]
        if self.verbose:
            print("Folders found:")
            print(self.subfolders)

    def load_metadata(self):
        """
        Loads metadata (but not image data) of LUNA16
        """
        if self.subfolders is None:
            self.find_subfolders(self.folder)
            if self.verbose:
                print("-" * 65)
                print("")
                print("From directory" + self.folder)
                print("Loading metadata from the following subfolders:")
                print(*self.subfolders, sep="\n")
                print("")
                print("-" * 65)
        for sub_folder_name in self.subfolders:
            curr_folder = self.folder.joinpath(sub_folder_name)
            files = [x for x in curr_folder.iterdir() if x.is_file()]
            files = [f for f in files if f.name.endswith(".mhd")]
            self.images.extend([LunaImage(curr_folder,f) for f in files]) # if os.path.splitext(f)[0] in nodulefiles # This lets only load images with nodules
        for i in self.images:
            i.unit="HU"   # I just know LUNA is in HUs
        self.load_nodule_metadata()

    def load_nodule_metadata(self,folder=None):
        """
        Loads the metadata of the LUNA16 nodule dataset and assigns it to the relevant image
        """
        if not self.images:
            raise AttributeError("Image metadata not loaded")

        if folder is None:
            folder=self.folder

        # Find nodule metadata
        nodule_file=folder.joinpath('annotations.csv')
        nodules=[]
        with open(nodule_file, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=' ')
            next(csvreader) # remove header
            for row in csvreader:
                nodules.append(Nodule(row[0]))
        # Obtain unique list of filenames
        if self.verbose:
            print(f"Nodules found: {len(nodules)}")

        # Now we need to assign the nodules to the images
        for n in nodules:
            for image in self.images:
                if n.filename == image.file_path.stem:
                    image.set_nodule_data(n)
                    break # if we found the image, break the loop, go to next nodule
