import os
import numpy as np

import h5py

from .load_utils import read_config_file, get_dataset_name


class LoadReader:

    def __init__(self, path: str, increment: int = 1):

        cfg = read_config_file(path)
        self.processing_dir = cfg["processing_dir"]
        self.datasets = cfg["datasets"]
        self.load_entry = cfg["load_entry"]
        self.projs_entry = cfg["proj_entry"]
        self.increment = increment
        self.time_entry = "/2.1/measurement/epoch_trig"

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

    def _validate_acquisition(self, path: str):

        with h5py.File(path, "r") as hin:

            if "2.1" in hin.keys():
                if hin[self.projs_entry][:].shape[0] > 600:
                    return True
                else:
                    return False
            else:
                if hin["/1.1/measurement/marana"][:].shape[0] > 600:
                    return True
                else:
                    return False

    def get_all_loads(self):

        times = []
        loads = []
        print(f"TODO: H5 entries are hardcoded. To be changed...")
        for selected_dataset in self.datasets:

            if not self._validate_acquisition(selected_dataset):
                continue

            try:

                with h5py.File(selected_dataset, "r") as hin:

                    if "2.1" in hin.keys():
                        time = hin["/2.1/measurement/epoch_trig"][:]
                        load = hin["/2.2/measurement/stress_adc_input"][:]

                    else:
                        time = hin["/1.1/measurement/epoch_trig"][:]
                        load = hin["/1.2/measurement/stress_adc_input"][:]
                times.append(np.mean(time))
                loads.append(np.mean(load))
            except Exception as e:
                time = 1e9
                load = 1e9

                times.append(time)
                loads.append(load)
        return np.array(times - times[0]), np.array(loads)

    def get_loading_curve(self):

        times = []
        loads = []
        print(f"TODO: H5 entries are hardcoded. To be changed...")
        for selected_dataset in self.selected_datasets:

            if not self._validate_acquisition(selected_dataset):
                continue

            with h5py.File(selected_dataset, "r") as hin:

                if "2.1" in hin.keys():
                    time = hin["/2.1/measurement/epoch_trig"][:]
                    load = hin["/2.2/measurement/stress_adc_input"][:]

                else:
                    time = hin["/1.1/measurement/epoch_trig"][:]
                    load = hin["/1.2/measurement/stress_adc_input"][:]
            times.append(np.mean(time))
            loads.append(np.mean(load))
        return np.array(times - times[0]), np.array(loads)
