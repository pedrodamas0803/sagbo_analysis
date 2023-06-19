import configparser
import os
from dataclasses import dataclass
from slurm_utils import read_config_file

@dataclass
class SlurmInfo:

    config_file:str
    partition_cpu: str = 'nice'
    partition_gpu: str = 'gpu'
    nodes: int = 6
    ntasks: int = 1
    cpus_per_task: int = 40
    mem_gb:int = 400
    time_cpu:str = '12:00:00'
    time_gpu:str = '12:00:00'
    output:str = 'insitu_%x.%j.out'
    error:str = 'insitu_%x.%j.err'

    

    

