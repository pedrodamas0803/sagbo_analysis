import os
import concurrent.futures
import time
import sys

import gmsh
import networkx
import numpy as np
import h5py
import corrct as cct
import scipy.ndimage as ndi
from nabu.preproc.phase import PaganinPhaseRetrieval
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave
from skimage.filters import threshold_otsu

from dvc_preprocessing.preprocessing import crop_around_CoM, volume_CoM
from .meshing_utils import read_config_file, get_dataset_name
from ..utils import calc_color_lims


class Meshing:
    def __init__(
        self,
        path: str,
        delta_beta=60,
        mesh_size: int = 12,
        reference_volume: int = 0,
        mult=1,
        slab_size=350,
        prop=0.25,
        iters=5,
    ):
        cfg = read_config_file(path=path)

        self.mesh_size = mesh_size
        self.processing_dir = cfg["processing_dir"]
        self.datasets = cfg["datasets"]
        self.overwrite = True if cfg["overwrite"] == True else False
        self.energy = float(cfg["energy"])
        self.distance_entry = cfg["distance_entry"]
        self.pixel_size_m = float(cfg["pixel_size_m"])
        self.delta_beta = float(delta_beta)
        self._reference_volume = reference_volume
        self.mesh_dir = os.path.join(self.processing_dir, "meshing")

        self.mult = mult
        self.slab_size = slab_size
        self.prop = prop
        self.iter = iters
        self.h5_path = os.path.join(
            self.mesh_dir, f"{get_dataset_name(self.selected_datasets)}.h5"
        )
        self.tiff_path = os.path.join(
            self.mesh_dir, f"{get_dataset_name(self.selected_datasets)}.tiff"
        )

        self._check_mesh_dir()

    @property
    def distance(self):
        with h5py.File(self.datasets[0], "r") as hin:
            distance = hin[self.distance_entry][()]
        return distance * 1e-3

    @property
    def selected_datasets(self):
        return self.datasets[self._reference_volume]

    @property
    def reference_data_path(self):
        name = get_dataset_name(self.selected_datasets)
        path_to_process = os.path.join(self.processing_dir, name, f"{name}.h5")
        proc_paths = path_to_process
        return proc_paths

    def _get_corr_projections(self):
        print("Will get projections from file.")

        with h5py.File(self.reference_data_path, "r") as hin:
            projs = hin["projections"][:].astype(np.float32)
            angles = hin["angles"][:].astype(np.float32)
            shifts = hin["shifts"][:]
        return projs, angles, shifts

    def _check_mesh_dir(self):
        if not os.path.exists(self.mesh_dir):
            os.mkdir(self.mesh_dir)
            print("Created mesh directory.")

    def _retrieve_phase(self):
        projs, angles, shifts = self._get_corr_projections()

        t0 = time.time()
        paganin = PaganinPhaseRetrieval(
            projs[0].shape,
            distance=self.distance,
            energy=self.energy,
            delta_beta=self.delta_beta,
            pixel_size=self.pixel_size_m,
        )

        ret_projs = np.zeros_like(projs)
        with concurrent.futures.ProcessPoolExecutor(os.cpu_count() - 2) as pool:
            for ii, proj in enumerate(pool.map(paganin.retrieve_phase, projs)):
                ret_projs[ii] = proj

        print(
            f"Applied phase retrieval on the stack of projections in {time.time()-t0}."
        )
        angles_rad = np.deg2rad(angles)
        self._save_projections(projs=ret_projs, angles=angles_rad, shifts=shifts)

        return np.rollaxis(ret_projs, 1, 0), angles_rad, shifts

    def _save_projections(
        self, projs: np.ndarray, angles: np.array, shifts: np.ndarray
    ):
        with h5py.File(self.h5_path, "w") as hout:
            hout["pag_proj"] = projs
            hout["angles"] = angles
            hout["delta_beta"] = self.delta_beta
            hout["shifts"] = shifts

        print("Saved data in the meshing folder")

    def _save_rec_vol(self, volume: np.ndarray):
        with h5py.File(self.h5_path, "a") as hout:
            hout["volFBP"] = volume
        print("Saved volume to h5 file.")

    def _reconstruct(self):
        print("Will start phase retrieval.")

        data_vwu, angles_rad, shifts = self._retrieve_phase()

        solverFBP = cct.solvers.FBP(fbp_filter="hann")
        proj_geom = cct.models.ProjectionGeometry.get_default_parallel()
        proj_geom.set_detector_shifts_vu(shifts)

        vol_geom = cct.models.VolumeGeometry.get_default_from_data(data_vwu)

        with cct.projectors.ProjectorUncorrected(
            vol_geom, angles_rad, prj_geom=proj_geom
        ) as A:
            volFBP, _ = solverFBP(A, data_vwu, iterations=10)

        print("Finished reconstruction.")

        self._save_rec_vol(volume=volFBP)

        return volFBP

    def _vol_post_processing(self, volume: np.ndarray):
        print("Will start post-processing of FBP volume")
        imin, imax = calc_color_lims(img=volume, mult=self.mult)

        center_of_mass = volume_CoM(image=volume, slab_size=self.slab_size)

        cropped_vol = crop_around_CoM(
            image=volume, CoM=center_of_mass, xprop=self.prop, yprop=self.prop
        )

        rescaled_vol = rescale_intensity(
            image=cropped_vol, in_range=(imin, imax), out_range=np.uint8
        )

        imsave(self.tiff_path, rescaled_vol, plugin="tifffile", check_contrast=False)

        print("Saved tiff volume.")
        del volume, cropped_vol

        return rescaled_vol

    def _create_mask(self, volume: np.ndarray = None):
        if volume is None:
            try:
                volume = imread(self.tiff_path, plugin="tifffile")
            except FileNotFoundError as e:
                print(e)
                sys.exit(1)
        tmp = np.zeros_like(volume)
        tmp[
            3 * self.mesh_size : -3 * self.mesh_size,
            3 * self.mesh_size : -3 * self.mesh_size,
            3 * self.mesh_size : -3 * self.mesh_size,
        ] = 1
        selem = np.ones((7, 7, 7), dtype=np.uint8)
        threshold = threshold_otsu(volume)
        mask = np.zeros_like(volume)
        whr = np.where(volume > threshold)
        mask[whr] = 1

        mask = ndi.binary_closing(mask, structure=selem, iterations=self.iter)
        mask = ndi.binary_opening(mask, structure=selem, iterations=self.iter)

        mask = ndi.binary_dilation(mask, structure=selem, iterations=self.iter)
        mask = ndi.binary_erosion(mask, structure=selem, iterations=self.iter).astype(
            np.uint8
        )
        mask *= tmp
        save_path = self.tiff_path.strip(".tiff") + "_mask.tiff"
        imsave(save_path, mask, plugin="tifffile")

        return mask
