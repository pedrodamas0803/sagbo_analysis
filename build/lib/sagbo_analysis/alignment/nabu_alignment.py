import os
import numpy as np

# import h5py

from nabu.reconstruction.projection import Projector
from nabu.reconstruction.fbp import Backprojector

from skimage.registration import phase_cross_correlation


class TomoAlignment:
    def __init__(
        self, projections: np.ndarray, angles: np.ndarray, upsample_factor: int = 5
    ) -> None:
        self.projs = projections
        self.shape = projections.shape
        self.angles = angles
        self.upsample_factor = upsample_factor
        self.init_shifts = self._initialize_shifts()
        self.init_cor = self.projs.shape[-1] / 2.0 + np.mean(self.init_shifts[1])
        self._cor = self.init_cor
        self._shifts = self.init_shifts

    def _initialize_shifts(self):
        _, ny, nx = self.shape

        shifts_vt = []
        shifts_ht = []

        for ii, _ in enumerate(self.projs):
            if ii < self.shape[0] // 2:
                im1 = self.projs[ii]
                im2 = np.fliplr(self.projs[ii + self.shape[0] // 2])
                (tv, th), _, _ = phase_cross_correlation(
                    im1, im2, upsample_factor=self.upsample_factor
                )

                shifts_vt.append(tv / 2.0)
                shifts_ht.append(th / 2.0)
        shifts_vt = np.array(shifts_vt)
        shifts_ht = np.array(shifts_ht)

        # shifts_v = np.concatenate([shifts_vt, np.flip(shifts_vt)])
        # shifts_h = np.concatenate([shifts_ht, np.flip(shifts_ht)])

        shifts_v = np.concatenate([shifts_vt, shifts_vt])
        shifts_h = np.concatenate([shifts_ht, shifts_ht])

        return np.stack([shifts_v, shifts_h])

    def _reconstruct(self):
        nz, ny, nx = self.shape

        data_vwu = np.rollaxis(self.projs, 1, 0)

        backproj = Backprojector(
            sino_shape=(nz, nx),
            angles=np.deg2rad(self.angles),
            filter_name="hann",
            rot_center=self._cor,
            extra_options={"axis_correction": self.init_shifts[1]},
        )

        x0 = np.zeros((ny, nx, nx), dtype=np.float32)
        for ii, sino in enumerate(data_vwu):
            slc = backproj.fbp(sino)
            x0[ii] = slc

        return x0

    def _project(self, rec_vol: np.ndarray):
        nz, ny, nx = self.shape

        radon = Projector(
            slice_shape=(nx, nx),
            angles=np.deg2rad(self.angles),
            rot_center=self._cor,
            detector_width=nx,
            extra_options={"axis_corrections": self.init_shifts[1]},
        )

        data_vwu = np.zeros((ny, nz, nx), dtype=np.float32)
        for ii, slc in enumerate(rec_vol):
            sino = radon.projection(image=slc)
            data_vwu[ii] = sino

        return np.rollaxis(data_vwu, 1, 0)

    def _correlate_projections(self, projs1: np.ndarray, projs2: np.ndarray):
        shifts_v = []
        shifts_h = []

        for old_proj, new_proj in zip(projs1, projs2):
            (tv, th), _, _ = phase_cross_correlation(
                old_proj, new_proj, upsample_factor=self.upsample_factor
            )

            shifts_v.append(tv)
            shifts_h.append(th)

        self._shifts = np.stack([shifts_v, shifts_h])

        return self._shifts


if __name__ == "__main__":
    import numpy as np

    import h5py
    import matplotlib.pyplot as plt
    from sagbo_analysis.alignment.nabu_alignment import TomoAlignment

    data_path = "/home/esrf/damasres/sagbo/g9_s5/g9_s5_sagbo_insitu_pct_0000/g9_s5_sagbo_insitu_pct_0000.h5"

    with h5py.File(data_path, "r") as hin:
        projs = hin["projections"][:, 900:1100, :]
        angles_rad = hin["angles"][:]

    f, (ax1, ax2) = plt.subplots(1, 2)

    plt.colorbar(ax1.imshow(projs[0], cmap="gray"))
    plt.colorbar(ax2.imshow(projs[:, 100], cmap="gray"))

    plt.show(block=False)

    align = TomoAlignment(projs, angles_rad)

    vol = align._reconstruct()

    f, (ax1, ax2) = plt.subplots(1, 2)

    plt.colorbar(ax1.imshow(vol[0], cmap="gray"))
    plt.colorbar(ax2.imshow(vol[:, :, 1024], cmap="gray"))

    plt.show(block=False)

    sinos = align._project(rec_vol=vol)

    f, (ax1, ax2) = plt.subplots(1, 2)

    plt.colorbar(ax1.imshow(sinos[0], cmap="gray"))
    plt.colorbar(ax2.imshow(sinos[:, 100], cmap="gray"))

    plt.show(block=False)

    f, (ax1, ax2) = plt.subplots(1, 2)

    plt.colorbar(ax1.imshow(projs[0] - sinos[0], cmap="gray"))
    plt.colorbar(ax2.imshow(projs[:, 100] - sinos[:, 100], cmap="gray"))

    plt.show(block=False)

    updt_shifts = align._correlate_projections(projs1=projs, projs2=sinos)

    f, axs = plt.subplots(2, 2)

    axs[0, 0].plot(align.angles, align.init_shifts[0])
    axs[0, 0].set_title("Old vertical shifts")
    axs[0, 1].plot(align.angles, align.init_shifts[1])
    axs[0, 1].set_title("Old horizontal shifts")

    axs[1, 0].plot(align.angles, updt_shifts[0])
    axs[1, 0].set_title("New vertical shifts")
    axs[1, 1].plot(align.angles, updt_shifts[1])
    axs[1, 1].set_title("New horizontal shifts")

    f.tight_layout()
    plt.show()
