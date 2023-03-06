import configparser
import os
from dataclasses import dataclass


@dataclass
class SampleInfo:

    '''
    Class that holds information about a sample in a SAGBO experiment for processing

    sample_dir: str - directory where the data for the referred sample is stored, 
                      e.g.: '/data/visitor/proposal/beamline/session/sample_name'

    base_name: str - text appended to the sample name to save the sequence of datasets,
                     e.g.: '/data/visitor/proposal/beamline/session/sample_name/sample_name_{base_name}'

    darks_path: str - path to the h5 file containing the darks for the sample

    processing_dir: str - path to the processing directory where the data will be saved

    pca_flat_file: str - path to the file containing the PCA decomposition of the flat field images for the experiment,
                        ideally inside the 'processing_dir'
    '''

    # acquisition information
    sample_dir: str
    base_name: str
    darks_path: str
    processing_dir: str
    flats_path: str

    flats_entry: str = '/1.1/measurement/marana'
    projs_entry: str = '/2.1/measurement/marana'
    darks_entry: str = '1.1/measurement/marana'
    angles_entry: str = '2.1/measurement/diffrz'
    load_entry: str = '1.2/measurement/stress_adc_input'

    @property
    def exp_name(self):
        return self.sample_dir.split('/')[3]

    @property
    def sample_name(self):
        return self.sample_dir.split('/')[-2]

    @property
    def pca_flat_file(self):
        return os.path.join(self.processing_dir, 'PCA_flats.h5')

    @property
    def config_file(self):
        return os.path.join(self.processing_dir, 'config.ini')

    @property
    def datasets(self):
        datasets = []
        for dataset in os.listdir(self.sample_dir):
            if self.sample_name in dataset and self.base_name in dataset:
                datasets.append(f'{self.sample_dir}{dataset}/{dataset}.h5')
        datasets.sort()
        return {f'path_{ii+1}': dataset for ii, dataset in enumerate(datasets)}

    def _generate_config(self):

        config = configparser.ConfigParser()

        config['DIRECTORIES'] = {
            'Acquisition_dir': self.sample_dir,
            'Base_name': self.base_name,
            'Darks_path': self.darks_path,
            'Flats_path': self.flats_path,
            'Processing_dir': self.processing_dir,
            'PCA_flat_file': self.pca_flat_file,
            'Experiment_name': self.exp_name,
            'Sample_name': self.sample_name,
            'Config_file': self.config_file
        }

        config['DATASETS'] = self.datasets

        # config['RECONSTRUCTION'] = {
        #     'Number_of_acquisitions' : 1,
        #     'Slab_size' : 256,
        #     'Do_SIRT' : False
        # }

        config['ENTRIES'] = {
            'Projections': self.projs_entry,
            'Flats': self.flats_entry,
            'Darks': self.darks_entry,
            'Angles': self.angles_entry,
            'Load': self.load_entry
        }

        return config

    def write_config(self):

        config = self._generate_config()
        if not os.path.exists(self.processing_dir):
            os.mkdir(self.processing_dir)

        self._create_processing_dirs()

        with open(self.config_file, 'w') as configfile:

            config.write(configfile)

    def _create_processing_dirs(self):

        for dataset in self.datasets.values():
            dataset_name = os.path.splitext(dataset)[0].split('/')[-1]
            print(dataset_name)

            dataset_proc_dir = os.path.join(self.processing_dir, dataset_name)

            if not os.path.exists(dataset_proc_dir):
                os.mkdir(dataset_proc_dir)
                print(f'Created {dataset_name} processing folder.')


# class SampleConfigWriter(SampleInfo):

#     def __init__(self, config:SampleInfo):
#         super().__init__()

#     def _generate_config(self):

#         config = configparser.ConfigParser()

#         config['DIRECTORIES'] = {
#             'Acquisition_directory' : self.sample_dir,
#             'Base_name' : self.base_name,
#             'Darks_path': self.darks_path,
#             'Processing_dir': self.processing_dir,
#             'PCA_flat_file': self.pca_flat_file,
#             'Experiment_name': self.exp_name,
#             'Sample_name': self.sample_name,
#             'Config_file': self.config_file
#         }

#         config['DATASETS'] = {
#             'datasets': self.datasets
#         }

#         # config['RECONSTRUCTION'] = {
#         #     'Number_of_acquisitions' : 1,
#         #     'Slab_size' : 256,
#         #     'Do_SIRT' : False
#         # }

#         config['ENTRIES'] = {
#             'Projections' : self.projs_entry,
#             'Flats' : self.flats_entry,
#             'Darks' : self.darks_entry
#         }

#         return config


#     def write_config(self):

#         config = self._generate_config()
#         if not os.path.exists(self.processing_dir):
#             os.mkdir(self.processing_dir)

#         with open(self.config_file, 'w') as configfile:

#             config.write(configfile)

# class SampleConfigReader:

#     def __init__(self, config_file:str):

#         cfg = _read_config(config_file)
#         sample_info = SampleInfo(
#             sample_dir = cfg['DIRECTORIES']['Acquisition_dir'],
#             base_name =  cfg['DIRECTORIES']['Base_name'],
#             darks_path = cfg['DIRECTORIES']['Darks_path'],
#             processing_dir = cfg['DIRECTORIES']['Processing_dir'],
#             pca_flat_file = cfg['DIRECTORIES']['PCA_flat_file'],
#             config_file = cfg['DIRECTORIES']['Config_file'],
#             flats_entry = cfg['ENTRIES']['Flats'],
#             projs_entry = cfg['ENTRIES']['Projections'],
#             darks_entry = cfg['ENTRIES']['Darks'],
#             angles_entry= cfg['ENTRIES']['Angles']
#         )

#         return sample_info


# def _read_config(path:str):
#     try:
#         config = configparser.ConfigParser()
#         config.read(path)
#     except FileNotFoundError as e:
#         print('The configuration file does not exist.')
