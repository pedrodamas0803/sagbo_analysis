import os
import concurrent.futures
import time
import sys
from typing import Any

import gmsh

import numpy as np
import h5py
import hdf5plugin
import corrct as cct
import scipy.ndimage as ndi
from nabu.preproc.phase import PaganinPhaseRetrieval
from numpy import ndarray, dtype
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave
from skimage.filters import threshold_otsu
from skimage.measure import marching_cubes
from skimage.morphology import binary_erosion, binary_dilation

from dvc_preprocessing.preprocessing import crop_around_CoM, volume_CoM
from .meshing_utils import (
    read_config_file,
    get_dataset_name,
    compute_loops,
    create_exterior_surfaces,
)
from ..utils import calc_color_lims


class Meshing:
    """
    Class to perform automatic meshing of a volume.
    It will gather the projections from the reference volume, apply a Paganin filter and reconstruct a volume. It then creates a mask that is used to run a marching cubes algorithm and subsequent meshing of the volume.

    """

    def __init__(
            self,
            path: str,
            delta_beta: int = 60,
            mesh_size: int = 12,
            reference_volume: int = 0,
            mult: float = 1,
            slab_size: int = 350,
            prop: float = 0.5,
            iters: int = 5,
            struct_size: tuple = (5, 5, 5),
            chunk_size: int = 512,
            redo: bool = False
    ):
        """
        __init__ initializes the class to perform meshing on the reference volume for a data-series.

        Parameters
        ----------
        path : str
            path to the configuration file.
        delta_beta : int, optional
            delta/beta value for the Paganin phase retrieval, by default 60
        mesh_size : int, optional
            mesh characteristc length, by default 12
        reference_volume : int, optional
            index of the volume to be used as reference volume, by default 0
        mult : float, optional
            selects how narrow the gray level distribution around the highest peak in the histogram in the form peak +/- mult/2 , by default 1
        slab_size : int, optional
            number of slices to be used for center of mass calculations , by default 350
        prop : float, optional
            fraction of the detector width used to cut the volume in post-processing, by default 0.5
        iters : int, optional
            number of iterations for binary erosion of the volume mask where the mesh will be calculated, by default 5
        struct_size : tuple, optional
            size of the structuring volume used in the marching cubes algorithm to determine the surface mesh of the reference volume, by default (5, 5, 5)
        chunk_size : int, optional
            number of slices to be reconstructed at once in a chunked reconstruction, by default 512
        """
        cfg = read_config_file(path=path)

        self.cfg_file = path
        self.mesh_size = mesh_size
        self.processing_dir = cfg["processing_dir"]
        self.datasets = cfg["datasets"]
        self.overwrite = True if cfg["overwrite"] == True else False
        self.energy = float(cfg["energy"])
        self.distance_entry = cfg["distance_entry"]
        self.pixel_size_m = float(cfg["pixel_size_m"])
        self.delta_beta = float(delta_beta)
        self._reference_volume = reference_volume
        self.mult = mult
        self.slab_size = slab_size
        self.prop = prop
        self.iter = iters
        self.struct_size = struct_size
        self.chunk_size = chunk_size
        self.mesh_dir = os.path.join(self.processing_dir, "meshing")
        self.h5_path = os.path.join(
            self.mesh_dir, f"{get_dataset_name(self.selected_datasets)}.h5"
        )
        self.tiff_path = os.path.join(
            self.mesh_dir, f"{get_dataset_name(self.selected_datasets)}.tiff"
        )
        self.surface_mesh = os.path.join(
            self.mesh_dir,
            f"{get_dataset_name(self.selected_datasets)}_surface_{self.mesh_size}.msh",
        )
        self.full_mesh = os.path.join(
            self.mesh_dir,
            f"{get_dataset_name(self.selected_datasets)}_full_{self.mesh_size}.msh",
        )
        self.mask_path = self.tiff_path.strip(".tiff") + "_mask.tiff"

        self.redo = True if redo else False
        #if redo:
        #     self._clean_meshing_dir(remove_h5=True)

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

        if os.path.exists(self.h5_path):
            with h5py.File(self.h5_path, "a") as hin:
                if "pag_proj" in hin.keys():
                    print(
                        "Phase retrieved projections already exist, loading from h5 file."
                    )
                    projs = hin["pag_proj"][:].astype(np.float32)
                    angles = hin["angles"][:]
                    shifts = hin["shifts"][:]
                # hin.close()
            is_retrieved = True
            return projs, angles, shifts, is_retrieved
        else:
            with h5py.File(self.reference_data_path, "r") as hin:
                print("Loaded projections to run phase retrieval algorithm.")
                projs = hin["projections"][:].astype(np.float32)
                angles = hin["angles"][:].astype(np.float32)
                shifts = hin["shifts"][:]
                # hin.close()
            is_retrieved = False
            return projs, angles, shifts, is_retrieved

    def _check_mesh_dir(self):
        if not os.path.exists(self.mesh_dir):
            os.mkdir(self.mesh_dir)
            print("Created mesh directory.")

    def _retrieve_phase(self):
        projs, angles, shifts, is_retrieved = self._get_corr_projections()

        t0 = time.time()
        if not is_retrieved:
            print(
                f'Will perform phase retrieval with pixel size {self.pixel_size_m:4} m, '
                f'propagation distance {self.distance:.4} m, at {self.energy} keV and delta/beta {self.delta_beta}. '
            )
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
                f"Applied phase retrieval on the stack of projections in {time.time() - t0}."
            )
            angles_rad = np.deg2rad(angles)
            self._save_projections(projs=ret_projs, angles=angles_rad, shifts=shifts)
            return np.rollaxis(ret_projs, 1, 0), angles_rad, shifts
        else:
            return np.rollaxis(projs, 1, 0), angles, shifts

    def _save_projections(
            self,
            projs: np.ndarray,
            angles: np.ndarray,
            shifts: np.ndarray,
    ):
        with h5py.File(self.h5_path, "a") as hout:
            hout["pag_proj"] = projs
            hout["angles"] = angles
            hout["delta_beta"] = self.delta_beta
            hout["shifts"] = shifts

            print("Saved data in the meshing folder")

    def _save_rec_vol(self, volume: np.ndarray):
        """

        Parameters
        ----------
        volume : reconstructed FBP volume
        """
        with h5py.File(self.h5_path, "a") as hout:
            if not 'volFBP' in hout.keys():
                hout["volFBP"] = volume
            else:
                hout['volFBP'][...] = volume
        print("Saved volume to h5 file.")

    def _reconstruct(self):
        '''
        Reconstructs the FBP volume in chunks of chunk_size length.

        Returns
        -------
        volFBP : the reconstructed FBP volume
        '''
        data_vwu, angles_rad, shifts = self._retrieve_phase()
        init_angle = angles_rad[0]
        angles_rad = angles_rad - init_angle

        solverFBP = cct.solvers.FBP(fbp_filter="hann")
        proj_geom = cct.models.ProjectionGeometry.get_default_parallel()
        proj_geom.set_detector_shifts_vu(shifts)

        n_subvols = data_vwu.shape[0] // self.chunk_size
        nz = data_vwu.shape[0]
        volFBP = np.zeros((nz, nz, nz), dtype=np.float32)

        for ii in range(n_subvols):
            zmin = ii * self.chunk_size
            zmax = (ii + 1) * self.chunk_size
            if zmax > data_vwu.shape[0]:
                zmax = data_vwu.shape[0]

            sub_data_vwu = data_vwu[zmin:zmax]
            vol_geom = cct.models.VolumeGeometry.get_default_from_data(sub_data_vwu)

            with cct.projectors.ProjectorUncorrected(
                    vol_geom, angles_rad, prj_geom=proj_geom
            ) as A:
                subvol, _ = solverFBP(A, sub_data_vwu, iterations=10)

            volFBP[zmin:zmax] = subvol

        print("Finished reconstruction.")

        self._save_rec_vol(volume=volFBP)

        return volFBP

    def _vol_post_processing(self, volume: np.ndarray):
        print("Will start post-processing of FBP volume")

        zmin = volume.shape[1] // 2 - self.slab_size // 2
        zmax = volume.shape[1] // 2 + self.slab_size // 2

        center_of_mass = volume_CoM(image=volume, init_slice=zmin, final_slice=zmax)

        cropped_vol = crop_around_CoM(
            image=volume, CoM=center_of_mass, xprop=self.prop, yprop=self.prop
        )
        imin, imax = calc_color_lims(img=cropped_vol, mult=self.mult)
        rescaled_vol = rescale_intensity(
            image=cropped_vol, in_range=(imin, imax), out_range=np.uint8
        )
        # rotate to match DCT/sample env reconstruction

        rotated_img = np.rot90(rescaled_vol, k=3, axes=(1, 2))

        imsave(self.tiff_path, rotated_img, plugin="tifffile", check_contrast=False)

        print("Saved tiff volume.")
        del volume, cropped_vol

        return rescaled_vol

    def _create_mask(self, volume: np.ndarray = None):
        if volume is None:
            try:
                volume = imread(self.tiff_path, plugin="tifffile")
                print("Imported volume from file.")
            except FileNotFoundError as e:
                print(e)
                sys.exit(1)
        tmp = np.zeros_like(volume)
        tmp[
        3 * self.mesh_size: -3 * self.mesh_size,
        3 * self.mesh_size: -3 * self.mesh_size,
        3 * self.mesh_size: -3 * self.mesh_size,
        ] = 1
        selem = np.ones(self.struct_size, dtype=np.uint8)
        threshold = threshold_otsu(volume)
        mask = np.zeros_like(volume)
        whr = np.where(volume > threshold)
        mask[whr] = 1

        mask = ndi.binary_closing(mask, structure=selem, iterations=self.iter // 2)
        mask = ndi.binary_opening(mask, structure=selem, iterations=self.iter // 2)

        # mask = ndi.binary_dilation(mask, structure=selem, iterations=self.iter)
        mask = ndi.binary_erosion(mask, structure=selem, iterations=self.iter).astype(
            np.uint8
        )
        mask *= tmp

        rotated_mask = np.rot90(mask, k=3, axes=(1, 2))

        imsave(self.mask_path, rotated_mask.astype(np.uint8), plugin="tifffile")

        return mask

    def _get_verts_and_triangles(self, volume: np.ndarray):
        vertices_s, triangles_s, _, _ = marching_cubes(
            volume,
            level=None,
            spacing=self.struct_size,
            step_size=self.mesh_size,
            allow_degenerate=False,
        )

        vertices_s = np.fliplr(vertices_s)
        triangles_s = np.fliplr(triangles_s)

        return vertices_s, triangles_s

    def _mesh_surface(self, vertices_s: np.ndarray, triangles_s: np.ndarray):
        gmsh.clear()
        gmsh.initialize()

        ### Import surface mesh to Gmsh model:
        vertices = vertices_s.copy()
        triangles = triangles_s.copy()

        # The tags of the nodes:
        ll = len(vertices)
        nodes = np.arange(1, ll + 1, 1).astype(np.uint32)

        # The connectivities of the triangle elements (3 node tags per triangle) with node tags starting at 1:
        triangles += 1

        # Create one discrete surface:
        stag = 1
        gmsh.model.addDiscreteEntity(2, stag)

        # gmsh.model.mesh.addNodes(dim, tag, nodeTags, coord, parametricCoord=[])
        gmsh.model.mesh.addNodes(2, stag, nodes, vertices.flatten())

        # gmsh.model.mesh.addElementsByType(tag, elementType, elementTags, nodeTags)
        # elementType 2 for 3-node triangle elements:
        gmsh.model.mesh.addElementsByType(stag, 2, [], triangles.flatten())

        gmsh.write(self.surface_mesh)

    def _mesh_volume(self):
        gmsh.clear()
        gmsh.initialize()
        gmsh.merge(self.surface_mesh)

        # reclassify the surface first, and compute the corresponding geometry:
        angle = 80.0
        forceParametrizablePatches = 1.0
        includeBoundary = True
        curveAngle = 180

        gmsh.model.mesh.classifySurfaces(
            np.deg2rad(angle),
            includeBoundary,
            forceParametrizablePatches,
            np.deg2rad(curveAngle),
        )
        gmsh.model.mesh.createGeometry()

        gmsh.model.geo.synchronize()

        # retrieve surface loops defining the boundaries of (closed) volumes:
        surfaces = gmsh.model.getEntities(2)
        lines = gmsh.model.getEntities(1)
        surfaceloops = compute_loops(surfaces, lines)

        print(surfaceloops)

        # the surfaces defining individual loops will be stored with Physical Surfaces numbered 10000 and above:
        looptags = []
        ii = 0
        for sloop in surfaceloops:
            gmsh.model.addPhysicalGroup(2, sloop, tag=10000 + ii)
            l = gmsh.model.geo.addSurfaceLoop(sloop)
            looptags.append(l)
            ii += 1

        # define the volume of the strut based on the exterior surface loop:
        v = gmsh.model.geo.addVolume(looptags, tag=1)
        gmsh.model.addPhysicalGroup(3, [v], tag=1)

        # now, define mesh size fields based on the strut exterior surface:
        struttags = surfaceloops[0]

        f1 = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f1, "SurfacesList", struttags)

        f2 = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f2, "InField", f1)
        gmsh.model.mesh.field.setNumber(f2, "SizeMin", 0.8 * self.mesh_size)
        gmsh.model.mesh.field.setNumber(f2, "SizeMax", 1.5 * self.mesh_size)
        gmsh.model.mesh.field.setNumber(f2, "DistMin", 0.8 * self.mesh_size)
        gmsh.model.mesh.field.setNumber(f2, "DistMax", 1.5 * self.mesh_size)

        gmsh.model.mesh.field.setAsBackgroundMesh(f2)

        gmsh.model.geo.synchronize()

        # A few meshing options before generating the 3D mesh:
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 5)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.Smoothing", 2)

        gmsh.model.mesh.generate(3)
        gmsh.write(self.full_mesh)
        gmsh.write(f"{self.full_mesh.strip('.msh')}.vtk")
        gmsh.clear()

    def mesh_volume(self):
        if os.path.exists(self.mask_path):
            mask = imread(self.mask_path, plugin="tifffile")
        elif os.path.exists(self.tiff_path):
            volume = imread(self.tiff_path, plugin="tifffile")
            mask = self._create_mask(volume=volume)
        elif os.path.exists(self.h5_path):
            try:
                with h5py.File(self.h5_path, "r") as hin:
                    volFBP = hin["volFBP"][:]
            except Exception as e:
                print(e, "\n", "FBP volume does not exist, will reconstruct")
                volFBP = self._reconstruct()

            volume = self._vol_post_processing(volume=volFBP)
            mask = self._create_mask(volume=volume)
        else:
            volFBP = self._reconstruct()
            volume = self._vol_post_processing(volume=volFBP)
            mask = self._create_mask(volume=volume)

        vertices, triangles = self._get_verts_and_triangles(volume=mask)

        self._mesh_surface(vertices_s=vertices, triangles_s=triangles)

        self._mesh_volume()

    def generate_mask(self):
        if not os.path.exists(self.h5_path) or self.redo:
            volFBP = self._reconstruct()
            print('Reconstructed your volume ! ')
        if not os.path.exists(self.tiff_path):
            volume = self._vol_post_processing(volume=volFBP)
            print('Finished post processing! ')
        else:
            print('Tiff volume already exists, loading it from file.')
            volume = imread(self.tiff_path, plugin = 'tifffile')
            print('Loaded tiff volume from file.')

        mask = np.zeros(volume.shape, dtype=bool)
        thrs = threshold_otsu(volume)
        whr = np.where(volume > thrs)
        mask[whr] = True
        print('Generated initial mask !')

        final_mask = np.zeros_like(mask)

        with concurrent.futures.ProcessPoolExecutor(os.cpu_count()-2) as pool:
            for ii, result in enumerate(pool.map(self.dilate_it, mask)):
                final_mask[ii] = result
        print('Finished binary dilation .')

        mask = np.zeros_like(mask)

        with concurrent.futures.ProcessPoolExecutor(os.cpu_count()-2) as pool:
            for ii, result in enumerate(pool.map(self.erode_it, final_mask)):
                mask[ii] = result
        print('Finished binary erosion')

        rescaled_mask = rescale_intensity(mask, in_range = (0, 1), out_range = np.uint8)

        rotated_mask = np.rot90(rescaled_mask, k=3, axes=(1, 2))

        imsave(self.mask_path, rotated_mask, plugin='tifffile')
        print('Saved mask ! ')

        return rotated_mask

    @staticmethod
    def dilate_it(slc: np.ndarray):
        return binary_dilation(slc, footprint=np.ones((25,25)))

    @staticmethod
    def erode_it(slc: np.ndarray):
        return binary_erosion(slc, footprint=np.ones((25, 25)))


    def _clean_meshing_dir(self, remove_h5: bool = False):
        for filename in os.listdir(self.mesh_dir):
            filepath = os.path.join(self.mesh_dir, filename)
            if remove_h5:
                os.remove(filepath)
            else:
                if filepath.endswith(".h5"):
                    print("Will keep the h5 file.")
                else:
                    os.remove(filepath)
        print("Cleaned the meshing directory.")
