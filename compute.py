#%% Import Modules
import os
from azureml.core import Experiment, Workspace, Run
from azureml.core.compute import ComputeTarget
from azureml.train.estimator import Estimator
from azureml.widgets import RunDetails
import shutil

def train(experiment_name, compute_target, payload):
    payload_folder = './payload'
    ws = get_workspace()
    exp = Experiment(workspace=ws, name=experiment_name)
    ct = ws.compute_targets[compute_target]

    ct.wait_for_completion(show_output=True)
    print(ct.status.serialize())

    ds = ws.get_default_datastore()
    print(ds.datastore_type, ds.account_name, ds.container_name)

    prepare_payload(payload_folder, payload)

    est = Estimator(
        source_directory=payload_folder,
        compute_target=ct,
        entry_script=payload[0])

    run = exp.submit(config=est)
    run
    run.wait_for_completion(show_output=True)
    print(run.get_metrics())

def upload(source, destination):
    ws = get_workspace()
    ds = ws.get_default_datastore()
    ds.upload(
        src_dir=source,
        target_path=destination,
        overwrite=False,
        show_progress=True)

def get_workspace():
    return Workspace.from_config(path='./config/workspace.json')

def prepare_payload(path, payload):
    os.makedirs(path, exist_ok=True)
    for file in payload:
        print('copying %s to payload folder' % file)
        shutil.copy(file, path)

#%%
train(
    experiment_name='style-transfer',
    compute_target='nv6',
    payload=['train.py'])

#%%
upload('./model', 'model')