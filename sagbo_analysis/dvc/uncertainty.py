import csv
import glob
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import skimage as sk

from .mscript import uncertainty_mesh_size, uncertainty_lambda_size, slurm_script
from .result import DVC_result
from .dvc_setup import DVC_Setup


class DVC_uncertainty(DVC_Setup):

    def __init__(self, config_file: str, increment: int = 1) -> None:
        super().__init__(config_file, increment)

    @property
    def ref_img_path(self):
        return f"{self.processing_paths[0].strip('.h5')}.tiff"

    @property
    def shifted_vol_path(self):
        return f"{os.path.splitext(self.ref_img_path)[0]}_shifted.tiff"

    @property
    def mesh_script_name(self):
        return os.path.join(self.uncertainty_dir, "mesh_uncertainty.m")

    @property
    def lambda_script_name(self):
        return os.path.join(self.uncertainty_dir, "lambda_uncertainty.m")

    def _import_reference_volume(self):
        vol = sk.io.imread(self.ref_img_path, plugin="tifffile")
        return vol

    @staticmethod
    def _generate_random_shifts():
        return np.random.random(size=3)

    def _clear_shifts(self):
        """
        Clears shifted volume and shifts file.
        """
        if os.path.exists(self.shifted_vol_path):
            os.remove(self.shifted_vol_path)

        files = os.listdir(self.uncertainty_dir)

        for file in files:
            if 'shift' in file:
                os.remove(file)

    @staticmethod
    def _shift_volume(vol: np.ndarray, shifts: tuple):

        """
        Shifts and returns the reference volume. Uses first order splines to
        interpolate values and ensure coherency with DVC code.
        """

        assert len(shifts) == vol.ndim

        dz, dy, dx = shifts

        shifted_vol = ndi.shift(vol, shift=(dz, dy, dx), order=1)

        return shifted_vol

    def _write_shifts(self, shifts: tuple):

        assert len(shifts) == 3

        dz, dy, dx = shifts

        d = datetime.now()

        wrt_path = os.path.join(
            self.uncertainty_dir,
            f"shifts_{d.year}{d.month:02}{d.day:02}{d.hour:02}h{d.minute:02}.txt",
        )

        with open(wrt_path, "w") as f:
            f.write(f"coord, shift\n")
            f.write(f"z, {dz}\n")
            f.write(f"y, {dy}\n")
            f.write(f"x, {dx}\n")

    def _save_shifted_vol(self, vol: np.ndarray):

        sk.io.imsave(self.shifted_vol_path, vol, plugin="tifffile")

    def prepare_uncertainty_analysis(self):

        self._clear_shifts()

        vol = self._import_reference_volume()

        print("Imported the reference volume.")

        shifts = self._generate_random_shifts()

        self._write_shifts(shifts=shifts)

        print("Generated and saved the shifts in a file.")

        shifted_vol = self._shift_volume(vol=vol, shifts=shifts)

        print("Shifted the volume.")

        self._save_shifted_vol(vol=shifted_vol)

        print("Saved the shifted volume.")

    def _get_roi(self, plot_image: bool = False):

        vol = self._import_reference_volume()

        nz, ny, nx = vol.shape

        flatened = np.max(vol, axis=0)

        thrs = sk.filters.threshold_otsu(flatened)

        mask = np.zeros(flatened.shape, np.uint8)

        mask[flatened >= thrs] = 255

        mask = sk.morphology.binary_erosion(mask, footprint=np.ones((150, 150)))

        mask = sk.morphology.binary_dilation(mask, footprint=np.ones((15, 15)))

        labeled = sk.measure.label(mask)

        props = sk.measure.regionprops(labeled)

        min_row, min_col, max_row, max_col = props[0].bbox

        min_depth = int(nz // 2 - nz // 4)

        max_depth = int(nz // 2 + nz // 4)

        if plot_image:
            plt.figure()
            plt.imshow(flatened, cmap="gray")
            plt.hlines([min_row, max_row], min_col, max_col)
            plt.vlines([min_col, max_col], min_row, max_row)
            plt.show()

        return (
            int(1.1 * min_col),
            int(0.9 * max_col),
            int(1.1 * min_row),
            int(0.9 * max_row),
            min_depth,
            max_depth,
        )

    def _write_mesh_script(self):

        roi = self._get_roi()
        script = uncertainty_mesh_size(
            ref_im=self.ref_img_path, def_im=self.shifted_vol_path, roi=roi
        )

        with open(self.mesh_script_name, "w") as f:
            for line in script:
                f.writelines(line)

        mscript = slurm_script(self.mesh_script_name.strip('.m'))

        with open('launch_mesh_unctty.slurm', 'w') as f:
            for line in mscript:
                f.writelines(line)

    def _write_lambda_script(self, mesh_size: int = 16):

        roi = self._get_roi()
        # mesh_size = self._guess_mesh_size()
        script = uncertainty_lambda_size(
            ref_im=self.ref_img_path,
            def_im=self.shifted_vol_path,
            mesh_size=mesh_size,
            roi=roi,
        )

        with open(self.lambda_script_name, "w") as f:
            for line in script:
                f.writelines(line)

        lscript = slurm_script(self.lambda_script_name.strip('.m'))
        with open('launch_lambda_unctty.slurm', 'w') as f:
            for line in lscript:
                f.writelines(line)

    @staticmethod
    def launch_slurm_script(which_script: str = 'mesh_size'):
        if which_script not in ['mesh_size', 'lambda_size']:
            print('Invalid script, try again!')
        elif which_script == 'mesh_size':
            os.system('sbatch launch_mesh_unctty.slurm')
        elif which_script == 'lambda_size':
            os.system('sbatch launch_lambda_unctty.slurm')


class DVC_uncertainty_summary(DVC_Setup):

    def __init__(self, config_file: str, increment: int = 1) -> None:
        super().__init__(config_file, increment)

        z_shift, y_shift, x_shift = self._get_shifts()

        self.zoffset = z_shift
        self.yoffset = y_shift
        self.xoffset = x_shift

        # self._create_results_folder()

    @property
    def results_folder(self):
        resfolder = os.path.join(self.uncertainty_dir, "results")
        if not os.path.exists(resfolder):
            os.mkdir(resfolder)
        return resfolder

    def _get_shift_files(self):
        shift_files = []
        for file in os.listdir(self.uncertainty_dir):
            if "shifts" in file and file.endswith(".txt"):
                shift_files.append(os.path.join(self.uncertainty_dir, file))
        return shift_files

    def _get_shifts(self):
        shift_file = self.choose_youngest_shift_file(self._get_shift_files())

        try:
            z_shift, y_shift, x_shift = self.parse_shift_file(shift_file)
        except ValueError as e:
            print(e)
            return -1

        return z_shift, y_shift, x_shift

    @staticmethod
    def parse_shift_file(path: str):
        with open(path, "r") as f:
            freader = csv.reader(f, delimiter=",")
            for row in freader:
                if row[0].strip() == "z":
                    z = float(row[1].strip())
                elif row[0].strip() == "y":
                    y = float(row[1].strip())
                elif row[0].strip() == "x":
                    x = float(row[1].strip())
                elif row[0].strip() == "coord":
                    continue
                else:
                    raise ValueError(
                        "The right flags were not found, restart the uncertainty processing."
                    )

        return z, y, x

    @staticmethod
    def choose_youngest_shift_file(filelist: list):
        highest_tt = -1
        highest_index = -1
        for ii, file in enumerate(filelist):
            tt = os.path.getmtime(file)
            if tt > highest_tt:
                highest_index = ii
                highest_tt = tt
        return filelist[highest_index]

    def _get_resfiles(self, analysis_type: str):

        if not analysis_type in ["mesh_size", "lambda"]:
            analysis_type = "mesh_size"

        resfiles = []
        if analysis_type == "mesh_size":
            resunclean = glob.glob(
                os.path.join(self.uncertainty_dir, "unctty_mesh_*.res")
            )
        else:
            resunclean = glob.glob(
                os.path.join(self.uncertainty_dir, "unctty_lambda_*.res")
            )
        for file in resunclean:
            if not "error" in file:
                resfiles.append(file)

        resfiles.sort()

        return resfiles

    def mesh_size_summary(self):

        results = self._get_results_dict(analysis_type="mesh_size")

        reg_par = results[0]["reg_par"]

        mesh_size = []
        std = []
        for result in results:
            mesh_size.append(result["mesh_size"])
            std.append(result["std"])
        mesh_size = np.array(mesh_size)
        std = np.array(std)

        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))

        ax.plot(mesh_size, std, "r+")
        # ax.set_ylim(ymax = 2.0)
        # ax.semilogx()
        ax.semilogy()
        if reg_par == 1000:
            ax.set_title(f"No regularization used.")
        else:
            ax.set_title(f"regularization = {reg_par}")

        fig.tight_layout()

        fig.savefig(
            os.path.join(self.results_folder, "mesh_size_uncertainty.png"),
            bbox_inches="tight",
            edgecolor="white",
            facecolor="white",
        )
        # choice = self.choose_mesh_size(mesh_size, std)
        # print(f"The lowest uncertainty level is at a mesh size = {choice}.")

    def lambda_size_summary(self):

        results = self._get_results_dict(analysis_type="lambda")

        mesh_size = results[0]["mesh_size"]

        reg_par = []
        std = []
        for result in results:
            reg_par.append(result["reg_par"])
            std.append(result["std"])
        reg_par = np.array(reg_par)
        std = np.array(std)

        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))

        ax.plot(reg_par, std, "r+")
        # ax.set_ylim(ymax = 2.0)
        # ax.semilogx()
        ax.semilogy()
        ax.set_title(f"Mesh size = {mesh_size}.")

        fig.tight_layout()

        fig.savefig(
            os.path.join(self.results_folder, "lambda_size_uncertainty.png"),
            bbox_inches="tight",
            edgecolor="white",
            facecolor="white",
        )

    def _get_results_dict(self, analysis_type: str):

        dict_list = []
        for file in self._get_resfiles(analysis_type=analysis_type):
            res = DVC_result(
                res_path=file,
                z_pix_offset=self.zoffset,
                y_pix_offset=self.yoffset,
                x_pix_offset=self.xoffset,
            )

            dict_list.append(
                {
                    "reg_par": res.regularization_parameter,
                    "mesh_size": res.mesh_size,
                    "std": res.std(),
                }
            )

        return dict_list

    @staticmethod
    def choose_mesh_size(mesh_size: list, std: list, max_std: float = 0.1):

        possible_size = []
        possible_std = []

        for size, dev in zip(mesh_size, std):
            if dev > max_std:
                continue
            elif dev <= max_std:
                possible_size.append(size)
                possible_std.append(dev)

        lower = np.min(possible_std)
        index = -1

        for ii, std in enumerate(possible_std):
            if std == lower:
                index = ii

        return possible_size[index]
