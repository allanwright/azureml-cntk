import json
import os
import shutil

from azureml.core import Experiment, Workspace, Run, ScriptRunConfig
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import RunConfiguration
from azureml.train.estimator import Estimator
from azureml.widgets import RunDetails

class EzAml:
    """
    EzAml class provides convenient methods for training machine learning
    models on local compute resources or using Azure Machine Learning remote
    compute targets.
    """

    supported_config_items = ['subscription_id', 'resource_group',
        'workspace_name', 'experiment_name', 'compute_target',
        'payload_path', 'payload', 'dependencies',
        'custom_docker_base_image', 'use_gpu', 'verbose_output',
        'download_outputs', 'download_path']
    
    def _dict_to_list(self, dict):
        return ['%s=%s' % (k, v) for k, v in dict.items()]
    
    def _download_from_run(self, config, run):
        for file in run.get_file_names():
            run.download_file(file, config['download_path'])

    def _get_config_from_file(self, config_path=''):
        if not config_path:
            config_path = './config.json'
        
        with open(config_path, 'r') as read_file:
            return json.load(read_file)
    
    def _get_merged_config(self, config=None, **kwargs):
        if not config:
            config = self._get_config_from_file()
        
        for name in EzAml.supported_config_items:
            if name in kwargs and kwargs[name]:
                config[name] = kwargs[name]
        
        return config

    def _get_workspace(self, config=None, subscription_id=None,
                        resource_group=None, workspace_name=None):
        config = self._get_merged_config(
            config, subscription_id=subscription_id,
            resource_group=resource_group, workspace_name=workspace_name)
        
        return Workspace(
            config['subscription_id'], config['resource_group'],
            config['workspace_name'])
    
    def _get_local_run_config(self, config, script_params):
        rc = RunConfiguration()
        rc.environment.python.user_managed_dependencies = True
        return ScriptRunConfig(
            source_directory=config['payload_path'],
            script=config['payload'][0],
            arguments=self._dict_to_list(script_params),
            run_config=rc)
    
    def _get_remote_run_config(self, workspace, config, script_params):
        ct = workspace.compute_targets[config['compute_target']]
        ct.wait_for_completion(show_output=config['verbose_output'])
        return Estimator(
            source_directory=config['payload_path'],
            compute_target=ct,
            entry_script=config['payload'][0],
            script_params=script_params,
            pip_packages=config['dependencies'],
            custom_docker_base_image=config['custom_docker_base_image'],
            use_gpu=config['use_gpu'])
    
    def _prepare_payload(self, path, payload):
        os.makedirs(path, exist_ok=True)
        for file in payload:
            print('Copying %s to %s' % (file, path))
            shutil.copy(file, path)
    
    def download_from_storage(self, source, destination, overwrite=False,
                              verbose_output=None):
        """
        """
        config = self._get_merged_config(verbose_output=verbose_output)
        ws = self._get_workspace(config)
        ds = ws.get_default_datastore()
        ds.download(destination, prefix=source, overwrite=overwrite)
    
    def train(self, config=None, subscription_id=None, resource_group=None,
              workspace_name=None, experiment_name=None, compute_target=None,
              payload_path=None, payload=None, dependencies=None,
              script_params=None, custom_docker_base_image=None,
              use_gpu=None, verbose_output=None, download_outputs=None,
              download_path=None):
        """
        """        
        config = self._get_merged_config(
            config, subscription_id=subscription_id,
            resource_group=resource_group, workspace_name=workspace_name,
            experiment_name=experiment_name, compute_target=compute_target,
            payload_path=payload_path, payload=payload,
            dependencies=dependencies, script_params=script_params,
            custom_docker_base_image=custom_docker_base_image, use_gpu=use_gpu,
            verbose_output=verbose_output, download_outputs=download_outputs,
            download_path=download_path)
        
        ws = self._get_workspace(config=config)
        exp = Experiment(workspace=ws, name=config['experiment_name'])
        self._prepare_payload(config['payload_path'], config['payload'])

        if config['compute_target'] == 'local':
            rc = self._get_local_run_config(config, script_params)
        else:
            ds = ws.get_default_datastore()
            script_params['--data-folder'] = ds.as_mount()            
            rc = self._get_remote_run_config(ws, config, script_params)
        
        run = exp.submit(rc)
        run
        run.wait_for_completion(show_output=config['verbose_output'])
        print(run.get_metrics())

        if config['download_outputs']:
            self._download_from_run(config, run)

    def upload_to_storage(self, source, destination, overwrite=False, verbose_output=None):
        """
        """
        config = self._get_merged_config(verbose_output=verbose_output)
        ws = self._get_workspace(config)
        ds = ws.get_default_datastore()
        ds.upload(
            src_dir=source,
            target_path=destination,
            overwrite=overwrite,
            show_progress=config['verbose_output'])