import os
import numpy as np

# import h5py

from nabu.reconstruction.projection import Projector
from nabu.reconstruction.fbp import Backprojector

from skimage.registration import phase_cross_correlation


class TomoConsistency:
    def __init__(
        self, projections: np.ndarray, angles: np.array, upsample_factor: int = 5
    ) -> None:
        self.projs = projections
        self.shape = projections.shape
        self.angles = angles
        self.upsample_factor = upsample_factor
        self.init_shifts = self._initialize_shifts()
        self.init_cor = self.projs.shape[-1] / 2.0 + np.mean(self.init_shifts[1])

    def _initialize_shifts(self):
        shifts_vt = []
        shifts_ht = []

        for ii, _ in enumerate(self.projs):
            if ii < self.shape[0] // 2:
                im1 = self.projs[ii]
                im2 = np.fliplr(self.projs[ii + self.shape[0] // 2])
                (tv, th), _, _ = phase_cross_correlation(
                    im1, im2, upsample_factor=self.upsample_factor
                )

                shifts_vt.append(tv)
                shifts_ht.append(th)
        shifts_vt = np.array(shifts_vt)
        shifts_ht = np.array(shifts_ht)

        shifts_v = np.concatenate([shifts_vt, np.flip(shifts_vt)])
        shifts_h = np.concatenate([shifts_ht, np.flip(shifts_ht)])

        return np.stack([shifts_v, shifts_h])

    def _x0(self):
        data_vwu = np.rollaxis(self.projs, 1, 0)

        nz, ny, nx = data_vwu.shape

        backproj = Backprojector(
            (nz, nx),
            angles=np.deg2rad(self.angles),
            rot_center=self.init_cor,
            extra_options={"axis_correction": self.init_shifts[1]},
        )

        x0 = np.zeros((nz, nx, nx), dtype=np.float32)
        for ii, sino in enumerate(data_vwu):
            slc = backproj.fbp(sino)
            x0[ii] = slc

        return x0
