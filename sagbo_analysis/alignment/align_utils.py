import configparser
import os

import numpy as np
import corrct as cct


def binning(data: np.ndarray):
    data_vwu = data.reshape(
        [data.shape[0], data.shape[1] // 2, 2, data.shape[2] // 2, 2]
    )
    data_vwu = data_vwu.mean(axis=(-3, -1))
    return data_vwu


def read_config_file(path: str):

    cfg = configparser.ConfigParser()
    cfg.read(path)

    cfg_dict = {
        # 'flats_path': cfg.get('DIRECTORIES', 'flats_path'),
        # 'darks_path': cfg.get('DIRECTORIES', 'darks_path'),
        # 'pca_flat_file': cfg.get('DIRECTORIES', 'pca_flat_file'),
        "processing_dir": cfg.get("DIRECTORIES", "processing_dir"),
        "flats_entry": cfg.get("ENTRIES", "flats"),
        "darks_entry": cfg.get("ENTRIES", "darks"),
        "projs_entry": cfg.get("ENTRIES", "projections"),
        "angles_entry": cfg.get("ENTRIES", "angles"),
        "datasets": [path for _, path in cfg.items("DATASETS")],
        "dering": cfg.get("FLAGS", "dering"),
        "overwrite": cfg.get("FLAGS", "overwrite"),
    }

    return cfg_dict


def get_dataset_name(path: str):

    return os.path.splitext(path)[0].split("/")[-1]


def align_standalone(
    projections: np.ndarray,
    angles_rad: np.array,
    iterations: int = 5,
    background: float = 0.1,
    binproj: bool = True,
    verbose: bool = True,
):
    if binproj:
        projs_bin = binning(projections)
        del projections
    else:
        projs_bin = projections.copy()
        del projections

    data_vwu = np.rollaxis(projs_bin, 1, 0)

    optim = cct.utils_align.OptimizeDetectorShifts(
        data_vwu,
        angles_rad,
        solver_cls=cct.solvers.FBP,
        solver_opts={},
        verbose=verbose,
    )

    pre_shifts_v = optim.pre_align_shifts_v()
    pre_shifts_u, cor = optim.pre_align_shifts_u(background=background, robust=True)

    pre_shifts_vu = np.stack([pre_shifts_v, pre_shifts_u + cor], axis=0)
    print(pre_shifts_vu)

    cor2 = optim.pre_cor_u_360()
    print(f"Center-of-rotation found using 360 redundancy: {cor2}")

    shifts, _ = optim.tomo_consistency_traditional(cor2, iterations=iterations)
    
    if binproj:
        return shifts * 2, cor2 * 2
    else:
        return shifts, cor2
