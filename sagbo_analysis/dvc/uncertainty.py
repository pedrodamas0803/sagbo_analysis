import os, glob, csv
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import skimage as sk
from .dvc_utils import read_config_file, get_dataset_name
from .setup import DVC_Setup
from .result import DVC_result
from .mscript import uncertainty_mesh_size, uncertainty_lambda_size


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

    def _generate_random_shifts(self):
        return np.random.random(size=3)

    def _shift_volume(self, vol: np.ndarray, shifts: tuple):

        assert len(shifts) == vol.ndim

        dz, dy, dx = shifts

        shifted_vol = ndi.shift(vol, shift=(dz, dy, dx))

        return shifted_vol

    def _write_shifts(self, shifts: tuple):

        assert len(shifts) == 3

        dz, dy, dx = shifts

        d = datetime.now()

        wrt_path = os.path.join(
            self.uncertainty_dir,
            f"shifts_{d.year}{d.month}{d.day}{d.hour}h{d.minute}.txt",
        )

        with open(wrt_path, "w") as f:
            f.write(f"coord, shift\n")
            f.write(f"z, {dz}\n")
            f.write(f"y, {dy}\n")
            f.write(f"x, {dx}\n")

    def _save_shifted_vol(self, vol: np.ndarray):

        sk.io.imsave(self.shifted_vol_path, vol, plugin="tifffile")

    def prepare_uncertainty_analysis(self):

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

        mask = sk.morphology.binary_erosion(mask, footprint=np.ones((15, 15)))

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
            int(1.2 * min_col),
            int(0.8 * max_col),
            int(1.2 * min_row),
            int(0.8 * max_row),
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

    def _write_lambda_script(self):

        roi = self._get_roi()
        mesh_size = self._guess_mesh_size()
        script = uncertainty_lambda_size(
            ref_im=self.ref_img_path,
            def_im=self.shifted_vol_path,
            mesh_size=mesh_size,
            roi=roi,
        )

        with open(self.lambda_script_name, "w") as f:
            for line in script:
                f.writelines(line)


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

    def _get_resfiles(self):
        resfiles = []
        resunclean = glob.glob(os.path.join(self.uncertainty_dir, "unctty_mesh_*.res"))
        for file in resunclean:
            if not "error" in file:
                resfiles.append(file)

        resfiles.sort()

        return resfiles

    def mesh_size_summary(self):

        results = self._get_results_dict()

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
        # ax.semilogy()
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
        choice = self.choose_mesh_size(mesh_size, std)
        print(f"The lowest uncertainty level is at a mesh size = {choice}.")

    def _get_results_dict(self):

        dict_list = []
        for file in self._get_resfiles():
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
