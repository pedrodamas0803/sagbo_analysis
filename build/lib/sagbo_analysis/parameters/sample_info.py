import configparser
import os
from dataclasses import dataclass


@dataclass
class SampleInfo:

    """
    Class that holds information about a sample in a SAGBO experiment for processing

    sample_dir: str - directory where the data for the referred sample is stored,
                      e.g.: '/data/visitor/proposal/beamline/session/sample_name'

    base_name: str - text appended to the sample name to save the sequence of datasets,
                     e.g.: '/data/visitor/proposal/beamline/session/sample_name/sample_name_{base_name}'

    darks_path: str - path to the h5 file containing the darks for the sample

    processing_dir: str - path to the processing directory where the data will be saved

    pca_flat_file: str - path to the file containing the PCA decomposition of the flat field images for the experiment,
                        ideally inside the 'processing_dir'
    """

    # acquisition information
    sample_dir: str
    base_name: str
    darks_path: str
    processing_dir: str
    flats_path: str
    overwrite: bool = True

    flats_entry: str = "/1.1/measurement/marana"
    projs_entry: str = "/1.1/measurement/marana"
    darks_entry: str = "/1.1/measurement/marana"
    angles_entry: str = "/1.1/measurement/diffrz"
    load_entry: str = "/1.2/measurement/stress_adc_input"
    distance_entry: str = "/1.1/instrument/positioners/nfdtx"

    energy: float = 65.35
    pixel_size_m: float = 0.63e-6

    dering: bool = False
    backend: str = "ASTRA"

    @property
    def exp_name(self):
        return self.sample_dir.split("/")[3]

    @property
    def sample_name(self):
        return self.sample_dir.split("/")[-2]

    @property
    def pca_flat_file(self):
        return os.path.join(self.processing_dir, "PCA_flats.h5")

    @property
    def config_file(self):
        return os.path.join(self.processing_dir, "config.ini")

    @property
    def datasets(self):
        datasets = []
        for dataset in os.listdir(self.sample_dir):
            if self.sample_name in dataset and self.base_name in dataset:
                datasets.append(f"{self.sample_dir}{dataset}/{dataset}.h5")
        datasets.sort()
        return {f"path_{ii+1}": dataset for ii, dataset in enumerate(datasets)}

    def _generate_config(self):
        """Generates the configuration file for the data processing"""

        config = configparser.ConfigParser()

        config["DIRECTORIES"] = {
            "Acquisition_dir": self.sample_dir,
            "Base_name": self.base_name,
            "Darks_path": self.darks_path,
            "Flats_path": self.flats_path,
            "Processing_dir": self.processing_dir,
            "PCA_flat_file": self.pca_flat_file,
            "Experiment_name": self.exp_name,
            "Sample_name": self.sample_name,
            "Config_file": self.config_file,
        }

        config["DATASETS"] = self.datasets

        config["FLAGS"] = {
            "dering": self.dering,
            "backend": self.backend,
            "overwrite": self.overwrite,
        }

        config["ENTRIES"] = {
            "Projections": self.projs_entry,
            "Flats": self.flats_entry,
            "Darks": self.darks_entry,
            "Angles": self.angles_entry,
            "Load": self.load_entry,
            "Distance": self.distance_entry,
        }

        config["PHASE"] = {"Energy": self.energy, "Pixel_size_m": self.pixel_size_m}

        return config

    def write_config(self):
        """Writes the configuration file at the given data processing directory."""

        config = self._generate_config()
        if not os.path.exists(self.processing_dir):
            os.mkdir(self.processing_dir)

        self._create_processing_dirs()

        with open(self.config_file, "w") as configfile:
            config.write(configfile)

    def _create_processing_dirs(self):
        """Creates the folders for each acquisition after checking if it exists or not."""

        for dataset in self.datasets.values():
            dataset_name = os.path.splitext(dataset)[0].split("/")[-1]
            print(dataset_name)

            dataset_proc_dir = os.path.join(self.processing_dir, dataset_name)

            if not os.path.exists(dataset_proc_dir):
                os.mkdir(dataset_proc_dir)
                print(f"Created {dataset_name} processing folder.")
