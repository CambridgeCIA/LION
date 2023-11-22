import argparse
from typing import Dict
from pathlib import Path
import shutil

from scipy.interpolate import interp1d
import imageio.v2 as imageio
import numpy as np
import json
from tqdm import tqdm


def format_slice_index(index: int) -> str:
    str_index = str(index)
    while len(str_index) < 5:
        str_index = "0" + str_index
    return f"slice{str_index}"

def format_mode_index(index:int) -> str:
    return f"mode{index}"

def write_settings(settings_out_file_path:Path):
    '''
    Writes the default settings to reconstruct the images from the original paper
    '''
    settings ={
        'detector':{
            'subsampling':2,
            'correction':[1,0],
            'n_pixels':1912,
            'pixel_physical_size_reference':0.0748,
            'pixel_shift': 0,
            'binning':1
            },
        'angular':{
            'n_angles':3600,
            'subsampling':1
        },
        'reconstruction':{
            'n_pixels':[512,512],
            'n_voxels':512,
            'SOD':431.019989,
            'SDD':529.000488
        },

    }


    ### DETECTOR CORRECTIONS
    settings['detector']['pixel_physical_size_corrected'] = settings['detector']['subsampling'] * settings['detector']['binning'] * settings['detector']['pixel_physical_size_reference']
    settings['detector']['n_pixels_subsampled'] = settings['detector']['n_pixels'] // settings['detector']['subsampling']
    settings['detector']['width'] = settings['detector']['pixel_physical_size_corrected'] * settings['detector']['n_pixels_subsampled']
    # Physical width of the field of view in the measurement plane via intersect theorem.
    settings['reconstruction']['FOV_width'] = settings['detector']['width'] * settings['reconstruction']['SOD'] / settings['reconstruction']['SDD']

    settings['reconstruction']['voxel_size'] = settings['reconstruction']['FOV_width'] / settings['reconstruction']['n_voxels']

    # Scaling the geometry accordingly.
    settings['reconstruction']['scale_factor'] = 1./settings['reconstruction']['voxel_size']
    settings['reconstruction']['scaled_SDD'] = settings['reconstruction']['SDD'] * settings['reconstruction']['scale_factor']
    settings['reconstruction']['scaled_SOD'] = settings['reconstruction']['SOD'] * settings['reconstruction']['scale_factor']

    settings['detector']['scaled_pixel_physical_size'] = settings['detector']['pixel_physical_size_corrected'] * settings['reconstruction']['scale_factor']

    with open(settings_out_file_path, 'w') as out_f:
        json.dump(settings, out_f, indent = 4)

def pre_process_slice(settings:Dict, slice_index:int, mode_index:int, raw_folder_path:Path, processed_folder_path:Path):
    """
    pre_process_slice applies the right sinogram correction as a pre_processing step to avoid doing it at runtime
    """
    ### Initialisation: formatting indices from (int) to (str) and defining the path to the mode's folder
    formatted_slice_index = format_slice_index(slice_index)
    formatted_mode = format_mode_index(mode_index)
    ### Existence check: pass if folder exists
    folder_savepath = processed_folder_path.joinpath(f'{formatted_slice_index}/{formatted_mode}')
    if folder_savepath.is_dir():
        print(f'{folder_savepath} already exists, passing...')
        return

    path_to_mode = raw_folder_path.joinpath(f'{formatted_slice_index}/{formatted_mode}')

    ### Loading the sinogram from the paths
    reconstruction = imageio.imread(path_to_mode.joinpath('reconstruction.tif'))
    sinogram = imageio.imread(path_to_mode.joinpath('sinogram.tif')).astype('float32')
    flat1 = imageio.imread(path_to_mode.joinpath('flat1.tif')).astype('float32')
    flat2 = imageio.imread(path_to_mode.joinpath('flat2.tif')).astype('float32')
    dark  = imageio.imread(path_to_mode.joinpath('dark.tif')).astype('float32')
    flat:np.ndarray = np.mean(np.array([ flat1, flat2 ]), axis=0 )

    ### Applying the detector subsampling correction and remove the last projection (which is equal to the first)
    detector_subsampling = settings['detector']['subsampling']
    sinogram = (sinogram[:,0::detector_subsampling]+sinogram[:,1::detector_subsampling])[:-1,:]
    dark = dark[0,0::detector_subsampling]+dark[0,1::detector_subsampling]
    flat = flat[0,0::detector_subsampling]+flat[0,1::detector_subsampling]
    # Subtract the dark field, devide by the flat field,
    # and take the negative log to linearize the sinogram according to the Beer-Lambert law.
    sinogram = sinogram - dark
    sinogram = sinogram/(flat-dark)

    ### Apply detector shift correction
    detector_correction = settings['detector']['correction']
    detector_pixel_physical_size = settings['detector']['pixel_physical_size_reference']

    if slice_index < 2830 or 5520 < slice_index < 5871:
        detector_shift = detector_correction[0] * detector_pixel_physical_size
    else:
        detector_shift = detector_correction[1] * detector_pixel_physical_size
    ## Apply detector shift
    detector_grid = np.arange(0, settings['detector']['n_pixels']//settings['detector']['subsampling']) * detector_pixel_physical_size
    detector_grid_shifted = detector_grid + detector_shift
    detector_grid_shift_corrected = interp1d(detector_grid, sinogram, kind='linear', bounds_error=False, fill_value='extrapolate') #type:ignore
    sinogram:np.ndarray = detector_grid_shift_corrected(detector_grid_shifted)

    sinogram = np.ascontiguousarray(sinogram)

    ### The sinogram, reconstruction and segmentation (if mode_index == 2) are ready to be saved

    folder_savepath.mkdir(exist_ok=True, parents=True)

    assert np.shape(sinogram) == (3600,956)
    np.save(folder_savepath.joinpath('sinogram.npy'), sinogram)
    np.save(folder_savepath.joinpath('reconstruction.npy'), reconstruction)
    if mode_index == 2:
        segmentation = np.array(imageio.imread(path_to_mode.joinpath('segmentation.tif')))
        np.save(folder_savepath.joinpath('segmentation.npy'), segmentation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_folder_path', required=True, help='The path to the folder where the processed data will be saved')
    parser.add_argument('--processed_folder_path', required=True, help='The path to the folder where the processed data will be saved')
    args = parser.parse_args()
    ### Unpack arguments
    raw_folder_path = Path(args.raw_folder_path)
    processed_folder_path = Path(args.processed_folder_path)
    ### Write scan settings file
    write_settings(processed_folder_path.joinpath('scan_settings.json'))
    ### Load the settings file
    settings = dict(json.load(open(processed_folder_path.joinpath('scan_settings.json'))))
    ### Iterate on the slice data
    for slice_index in tqdm(range(1,5001)):
        for mode_index in range(1,4):
            pre_process_slice(
                settings,
                slice_index,
                mode_index,
                raw_folder_path,
                processed_folder_path
                )
    ### Copy the data records to the processed folder
    current_folder_path = Path(__file__).resolve().parent
    shutil.copy(
        current_folder_path.joinpath('default_data_records.csv'),
        processed_folder_path.joinpath('default_data_records.csv')
        )


