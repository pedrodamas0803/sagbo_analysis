import configparser
import os, time, concurrent.futures
import corrct as cct
import numpy as np
from nabu.preproc.phase import PaganinPhaseRetrieval


def read_config_file(path: str):
    """
    read_config_file gets relevant information from the config file.

    Parameters
    ----------
    path : str
        path to the config file.

    Returns
    -------
    dict
        dictionary containg the information for the recnstruction step.
    """

    cfg = configparser.ConfigParser()
    cfg.read(path)

    cfg_dict = {
        "processing_dir": cfg.get("DIRECTORIES", "processing_dir"),
        "datasets": [path for _, path in cfg.items("DATASETS")],
        # 'dering' : cfg.get('FLAGS', 'dering'),
        "overwrite": cfg.get("FLAGS", "overwrite"),
    }

    return cfg_dict


def get_dataset_name(path: str):
    return os.path.splitext(path)[0].split("/")[-1]


def FBP_reconstruction(
    sinograms: np.ndarray, angles_rad: np.array, shifts: np.ndarray
) -> np.ndarray:
    """
    _reconstruct_FBP reconstructs a tomography volume using a Filtered Back-Projection

    Parameters
    ----------
    sinograms : np.ndarray
        stack of sinograms to be used for the reconstruction
    angles_rad : np.array
        array of angles corresponding to each line of a sinogram
    shifts : np.ndarray
        array of detector shifts calculated using tomogrpahic consistency

    Returns
    -------
    volFBP : np.ndarray
        reconstructed volume
    """
    ang0 = angles_rad[0]
    angles_rad = angles_rad - ang0

    solverFBP = cct.solvers.FBP(verbose=False, fbp_filter="hann")
    vol_geom = cct.models.VolumeGeometry.get_default_from_data(sinograms)
    proj_geom = cct.models.ProjectionGeometry.get_default_parallel()
    proj_geom.set_detector_shifts_vu(shifts)

    with cct.projectors.ProjectorUncorrected(
        vol_geom, angles_rot_rad=angles_rad, prj_geom=proj_geom
    ) as A:
        volFBP, _ = solverFBP(A, sinograms, iterations=10)

    return volFBP

def paganin_retrieve_phase(projs, distance, energy, delta_beta, pixel_size_m):

        t0 = time.time()
        print(
            f"Will perform phase retrieval with pixel size {pixel_size_m:4} m, "
            f"propagation distance {distance:.4} m, at {energy} keV and delta/beta {delta_beta}. "
        )
        paganin = PaganinPhaseRetrieval(
            projs[0].shape,
            distance=distance,
            energy=energy,
            delta_beta=delta_beta,
            pixel_size=pixel_size_m,
        )

        ret_projs = np.zeros_like(projs)
        with concurrent.futures.ProcessPoolExecutor(os.cpu_count() - 2) as pool:
            for ii, proj in enumerate(pool.map(paganin.retrieve_phase, projs)):
                ret_projs[ii] = proj

        print(
            f"Applied phase retrieval on the stack of projections in {time.time() - t0}."
        )
        return np.rollaxis(ret_projs, 1, 0)