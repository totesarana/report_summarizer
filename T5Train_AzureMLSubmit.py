"""
Script to Run LungCAD Candidate Generator Training on AzureML
"""
import datetime
import os
import pathlib

from ailab.authentication import DefaultAilabCredential
from azure.ai.ml import MLClient
from azure.ai.ml import UserIdentityConfiguration
from azure.ai.ml import command, Input, Output
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import Environment, BuildContext

""" ============= Parameters ============= """
year_week = datetime.datetime.now().strftime("_%Y%U")

""" ============= Login to workspace ============= """
if os.name == 'nt':  # if windows machine
    credential = DefaultAilabCredential()
    ##
    ml_client = MLClient(credential, subscription_id='dcd361a2-6cb1-4371-b56a-b23024c90a1f',
                         resource_group_name="ailab-rg",
                         workspace_name="ailab")
else:
    ml_client = MLClient.from_config()
print(ml_client)

""" ============= EDIT BELOW THIS LINE ============= """


def run(computeTarget="V100xHigh", batchSize=8):
    pass


def runT5TrainingOnAzureML(computeTarget="ND40x8V100low", args=[
    '--GTjson', "F:/SmartReport/GT.json",
    '-batch_size', '8',
    '--epoches', '8',
    '--save_dir', 'F:/SmartReport/Training_snapshots',
    '--lr', '5.6e-5',
], seed=None, tags=None):
    if not tags:
        tags = dict()
    if seed:
        args += ["--seed", seed]
        tags["Seed"] = seed
    """ ============= Environment Selection/Creation ============= """
    try:
        version = 16  # 2
        myenv = ml_client.environments.get(name='LUNGCAD_VD20D_CG', version=version)
    except Exception:
        dockerfilepath = os.path.realpath(
            os.path.join(os.path.basename(__file__), "../../docker/mldockerfiles/lungcad/LUNGCAD_VD20D_CG"))
        buildContext = BuildContext(path=dockerfilepath, dockerfile_path="LUNGCAD_VD20D_CG.dockerfile")
        myenv = Environment(name='LUNGCAD_VD20D_CG', version=version, build=buildContext)
        ml_client.environments.create_or_update(myenv)
    """ ============= Define FDrive IN/OUT ============= """
    my_job_inputs = {
        "fdrivein": Input(type=AssetTypes.URI_FOLDER,
                          path='azureml://subscriptions/dcd361a2-6cb1-4371-b56a-b23024c90a1f/resourcegroups/ailab-rg/workspaces/ailab/datastores/fdrive/paths/',
                          mode=InputOutputModes.MOUNT),
    }
    my_job_outputs = {
        "fdriveout": Output(type=AssetTypes.URI_FOLDER,
                            path='azureml://subscriptions/dcd361a2-6cb1-4371-b56a-b23024c90a1f/resourcegroups/ailab-rg/workspaces/ailab/datastores/fdrive/paths/',
                            mode=InputOutputModes.MOUNT),
    }
    args += ['--fdriveinput', "${{inputs.fdrivein}}",
             '--fdriveoutput', "${{outputs.fdriveout}}"]
    """ ============= END Define FDrive IN/OUT ============= """
    commandline = "python T5_train_main.py " + " ".join(args)
    environmentvariables = dict()
    environmentvariables["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environmentvariables["PYTHONHASHSEED"] = "42"
    environmentvariables["AZ_EXPERIMENT"] = "TRUE"
    environmentvariables["AZUREML_COMPUTE_USE_COMMON_RUNTIME"] = "true"
    environmentvariables["RSLEX_DIRECT_VOLUME_WRITABLE_MOUNT"] = "true"
    if version > 3:  # starting from Pytorch 2.0
        environmentvariables['NCCL_NET'] = 'Socket'
    job = command(
        code=pathlib.Path(__file__).parent.resolve(),
        command=commandline,
        shm_size='64g',
        inputs=my_job_inputs,
        outputs=my_job_outputs,
        environment=f"{myenv.name}:{myenv.version}",
        environment_variables=environmentvariables,
        compute=computeTarget,
        identity=UserIdentityConfiguration(),
        experiment_name=f'SmartReport_T5{year_week}',
        tags=tags
    )

    """ ============= Specify Source Code Directory, Script and Compute ============= """

    # kcc = KubernetesComputeConfiguration()
    # kcc.instance_type = "gb16c6"
    # runConfig.kubernetescompute = kcc

    returned_job = ml_client.jobs.create_or_update(job)
    # get a URL for the status of the job
    print(returned_job.services["Studio"].endpoint)


if __name__ == "__main__":
    runT5TrainingOnAzureML(computeTarget="ND40x8V100low", seed="42", args=[
        '--GTjson', "F:/SmartReport/GT.json",
        '--batch_size', '8',
        '--epoches', '8',
        '--save_dir', 'F:/SmartReport/Training_snapshots',
        '--lr', '5.6e-5',
    ])