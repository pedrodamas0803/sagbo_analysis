import os, glob
from .dvc_utils import read_config_file, get_dataset_name


class DVC_Setup:
    def __init__(self, config_file: str, increment: int = 1) -> None:
        cfg = read_config_file(path=config_file)

        self.processing_dir = cfg["processing_dir"]
        self.datasets = cfg["datasets"]
        self.increment = increment

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

    def build_folder_structure(self):
        try:
            os.mkdir(self.dvc_dir)
        except FileExistsError:
            print("DVC directory already exists, skipping.")
        self._link_vtks()
        self._link_images()

    def _link_vtks(self):
        vtks = glob.glob(os.path.join(self.meshing_dir, "*.vtk"))

        for vtk in vtks:
            dst = os.path.join(self.dvc_dir, os.path.basename(vtk))
            os.symlink(src=vtk, dst=dst)

    def _link_images(self):
        for dataset in self.processing_paths:
            tiff_name = f"{dataset.strip('.h5')}.tiff"
            dst = os.path.join(self.dvc_dir, os.path.basename(tiff_name))
            os.symlink(src=tiff_name, dst=dst)
