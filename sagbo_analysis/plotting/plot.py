import os, h5py
from pkgutil import get_data
import numpy as np
import matplotlib.pyplot as plt

from ..utils import calc_color_lims
from .plot_utils import read_config_file, get_dataset_name


class SampleImagePlot:

    def __init__(self, config_file: str, increment: int = 1) -> None:

        cfg = read_config_file(path=config_file)

        self.processing_dir = cfg["processing_dir"]
        self.datasets = cfg["datasets"]
        self.increment = increment

        if not os.path.exists(self.images):
            os.mkdir(self.images)

    @property
    def selected_datasets(self):
        datasets = []
        for ii, dataset in enumerate(self.datasets):
            if ii % self.increment == 0:
                datasets.append(dataset)
        return datasets

    @property
    def processing_paths(self):
        proc_paths = []
        for dataset in self.selected_datasets:
            name = get_dataset_name(dataset)
            path_to_process = os.path.join(self.processing_dir, name, f"{name}.h5")
            proc_paths.append(path_to_process)
        return proc_paths

    @property
    def dvc_dir(self):
        return os.path.join(self.processing_dir, "DVC_Analysis")

    @property
    def meshing_dir(self):
        return os.path.join(self.processing_dir, "meshing")

    @property
    def images(self):
        return os.path.join(self.processing_dir, "images")

    # @property
    # def uncertainty_dir(self):
    #     return os.path.join(self.dvc_dir, "uncertainty")

    def _get_projection(self, path: str):

        sample_name = get_dataset_name(path)

        with h5py.File(path, "r") as hin:
            shape = hin["projections"].shape
            proj = hin["projections"][shape[0] // 2]
        return proj, sample_name

    def get_projections(self, ncols: int = 4, save=True):

        do_save = False
        all_names = []
        all_projs = []

        for dataset in self.processing_paths:
            proj, sample_name = self._get_projection(dataset)
            all_projs.append(proj)
            all_names.append(sample_name)

        if save:
            do_save = True
        self._save_image_grid(
            fn="projection_grid",
            images=all_projs,
            names=all_names,
            ncols=ncols,
            save=do_save,
        )

    def _save_image_grid(
        self,
        fn: str,
        images: np.ndarray,
        names: list = [],
        ncols: int = 4,
        save: bool = True,
    ):

        nrows = int(np.ceil(len(images) / ncols))

        f, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(16, 16))

        axs = axs.ravel()

        for ii, img in enumerate(images):
            imin, imax = calc_color_lims(img, int=2)

            axs[ii].imshow(img, vmin=imin, vmax=imax, cmap="gray")
            axs[ii].set_axis_off()
            if len(names) > 0:
                axs[ii].set_title(f"{names[ii]}")

        f.tight_layout()

        if save:
            f.savefig(
                os.path.join(self.images, f"{fn}.png"), facecolor="white", dpi=300
            )

    def _get_orthoslices(self, path: str):
        """
        Returns orthogonal slices in the order xy, xz, yz
        """

        sample_name = get_dataset_name(path)

        with h5py.File(path, "r") as hin:
            keys = list(hin.keys())
            if "volPDHG" in keys:
                tag = "volPDHG"
            elif "volSIRT" in keys:
                tag = "volSIRT"
            elif "volFBP" in keys:
                tag = "volFBP"
            else:
                tag = None
                print("No reconstruction was found. Skipping. ")
                return
            shape = hin[tag].shape
            xy = hin[tag][shape[0] // 2, :, :]
            xz = hin[tag][:, shape[1] // 2, :]
            yz = hin[tag][:, :, shape[2] // 2]

        return [xy, xz, yz], sample_name

    def get_orthogonal_views(self, save: bool = True):

        for file in self.processing_paths:

            orthoslices, sample_name = self._get_orthoslices(path=file)

            f, axs = plt.subplots(1, 3)

            axs = axs.ravel()

            for ax, orthoslice in zip(axs, orthoslices):
                imin, imax = calc_color_lims(orthoslice, int=2)
                ax.imshow(orthoslice, vmin=imin, vmax=imax, cmap="gray")

            f.suptitle(sample_name)

            f.tight_layout()
            if save:
                f.savefig(
                    os.path.join(self.images, f"{sample_name}.png"),
                    facecolor="white",
                    dpi=300,
                )
