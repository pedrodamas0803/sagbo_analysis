import os, concurrent.futures

import h5py
import numpy as np
from dvc_preprocessing.preprocessing import crop_around_CoM, volume_CoM
from skimage.exposure import rescale_intensity
from skimage.io import imsave
from skimage.morphology import binary_erosion, binary_dilation
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

from ..utils import calc_color_lims
from .postproc_utils import (
    build_tiff_path,
    get_dataset_name,
    read_config_file,
    build_mask_path,
)

NTHREAD = os.cpu_count() - 2


class PostProcessing:
    def __init__(
        self,
        path: str,
        increment: int = 1,
        acq_numbers: list = None,
        prop: float = 0.5,
        mult: float = 1,
        struct_size: int = 20,
        # test:bool = False
    ):
        cfg = read_config_file(path)

        self.processing_dir = cfg["processing_dir"]
        self.datasets = cfg["datasets"]
        self.prop = prop
        self.mult = mult
        self.struct_size = struct_size
        self.acq_numbers = acq_numbers
        if self.acq_numbers is not None:
            self.increment = 100000
        else:
            self.increment = increment
        # self.is_test = test

        # self.min32, self.max32 = self._calc_lims_32bit()

    @property
    def selected_datasets(self):
        datasets = []
        if self.acq_numbers is None:
            for ii, dataset in enumerate(self.datasets):
                if ii % self.increment == 0:
                    datasets.append(dataset)
        else:
            for ii, acq_number in enumerate(self.acq_numbers):
                datasets.append(self.datasets[acq_number])
        return datasets

    @property
    def processing_paths(self):
        proc_paths = []
        for dataset in self.selected_datasets:
            name = get_dataset_name(dataset)
            path_to_process = os.path.join(self.processing_dir, name, f"{name}.h5")
            proc_paths.append(path_to_process)
        return proc_paths

    # def _calc_lims_32bit(self):
    #     vol = self._load_volume(path=self.processing_paths[0])

    #     imin, imax = calc_color_lims(vol, mult=self.mult)

    #     return imin, imax

    def run_postprocessing(self, plot: bool = False, test: bool = False):
        for ii, dataset in enumerate(self.processing_paths):
            print(f"Processing {dataset}.")
            try:
                vol = self._load_volume(path=dataset)
            except Exception as e:
                print(e)
                print("There was a problem loading your volume, skipping to the next.")
                continue

            center_of_mass = volume_CoM(vol)

            cropped_vol = crop_around_CoM(
                vol, center_of_mass, xprop=self.prop, yprop=self.prop
            )

            imin, imax = calc_color_lims(cropped_vol, mult=self.mult)

            rescaled_img = rescale_intensity(
                cropped_vol, in_range=(imin, imax), out_range="uint8"
            )

            # rotate to match DCT/sample env reconstruction
            rotated_img = np.rot90(rescaled_img, k=3, axes=(1, 2))

            mask = self._calculate_mask(vol=rotated_img)

            if plot:
                self.plot_im(mask)

            if ii == 0:
                imsave(
                    build_mask_path(path=dataset),
                    mask,
                    plugin="tifffile",
                    check_contrast=False,
                )

            final_image = mask * rotated_img

            if test:
                print(f"Image dimensions: {final_image.shape}")
                print(
                    "This was a test. If you are satisfied, set the flag to False and re-run"
                )
                break

            save_path = build_tiff_path(dataset)

            imsave(save_path, final_image, plugin="tifffile", check_contrast=False)

    def _load_volume(self, path: str):
        with h5py.File(path, "r") as hin:
            if "volPDHG" in hin.keys():
                vol = hin["volPDHG"][:].astype(np.float32)
            elif "volSIRT" in hin.keys():
                vol = hin["volSIRT"][:].astype(np.float32)
            else:
                vol = hin["volFBP"][:].astype(np.float32)
        return vol

    def _calculate_mask(self, vol: np.ndarray):

        mask = np.zeros_like(vol)
        tmp = np.zeros_like(vol)

        thrs = threshold_otsu(vol)

        mask[vol > thrs] = np.iinfo(mask.dtype).max

        with concurrent.futures.ProcessPoolExecutor() as pool:

            for ii, result in enumerate(pool.map(self.dilate_it, mask)):
                tmp[ii] = result

        with concurrent.futures.ProcessPoolExecutor() as pool:

            for ii, result in enumerate(pool.map(self.erode_it, tmp)):
                mask[ii] = result

        return mask

    @staticmethod
    def plot_im(mask: np.ndarray):

        assert mask.ndim == 3

        f, axs = plt.subplots(1, mask.ndim)

        axs[0].imshow(mask[mask.shape[0] // 2, :, :], cmap="gray")
        axs[1].imshow(mask[:, mask.shape[1] // 2, :], cmap="gray")
        axs[2].imshow(mask[:, :, mask.shape[2] // 2], cmap="gray")

        f.tight_layout()
        plt.show()

    @staticmethod
    def erode_it(slc: np.ndarray):
        return binary_erosion(slc, np.ones((25, 25)))

    @staticmethod
    def dilate_it(slc: np.ndarray):
        return binary_dilation(slc, np.ones((25, 25)))
