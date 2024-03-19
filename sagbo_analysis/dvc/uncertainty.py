import os, glob, csv
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import skimage as sk
from .dvc_utils import read_config_file, get_dataset_name
from .setup import DVC_Setup
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

        mask = sk.morphology.binary_erosion(mask, footprint=np.ones((5, 5)))

        mask = sk.morphology.binary_dilation(mask, footprint=np.ones((5, 5)))

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
        script = uncertainty_lambda_size(
            ref_im=self.ref_img_path, def_im=self.shifted_vol_path, roi=roi
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
