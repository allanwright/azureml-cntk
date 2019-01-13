import argparse
import cntk as C
import numpy as np
from azureml.core.run import Run

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', default='./')
args = parser.parse_args()

run = Run.get_context()
run.log('cntk', C.__version__)
run.log('numpy', np.__version__)
run.log('data-folder', args.data_folder)