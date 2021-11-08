# -----------------------------------------------------
# This is the Job Script/Run Configuration script for 
# building a pipeline and running it in an experiment
# -----------------------------------------------------

from azureml.core import Workspace

# Access the Workspace
ws = Workspace.from_config("./config")



# -------------------------------------------------
# Create custom environment
from azureml.core import Environment
from azureml.core.environment import CondaDependencies

# Create the environment
pipeline_env = Environment(name="pipeline_env")

# install packages
#-----------------
conda_packages = CondaDependencies()
conda_packages.add_pip_package("azureml-sdk")
conda_packages.add_pip_package("pandas")
conda_packages.add_pip_package("sklearn")
# include package in the pipline_env
pipeline_env.python.conda_dependencies = conda_packages
# set user_manager_dependencies to True
pipeline_env.python.user_managed_dependencies = False
# register the envirnmont
pipeline_env.register(ws)
# ---------------------------------------------------



# ---------------------------------------------------
# Create a compute cluster for pipeline
# ---------------------------------------------------
cluster_name = "pipeline-cluster"

from azureml.core.compute import AmlCompute
compute_config = AmlCompute.provisioning_configuration(
                                    vm_size='STANDARD_D11_V2', 
                                    max_nodes=2)


from azureml.core.compute import ComputeTarget
compute_cluster = ComputeTarget.create(ws, cluster_name, compute_config)

compute_cluster.wait_for_completion()
# ---------------------------------------------------

# ---------------------------------------------------
# Create Run Configurations for the steps
from azureml.core.runconfig import RunConfiguration
run_config = RunConfiguration()

run_config.target = compute_cluster
run_config.environment = pipeline_env

# ---------------------------------------------------


# Define Pipeline steps
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core  import PipelineData

input_ds = ws.datasets.get('Defaults')

dataFolder = PipelineData('datafolder', datastore=ws.get_default_datastore())


# Step 01 - Data Preparation
dataPrep_step = PythonScriptStep(name='01 Data Preparation',
                                 source_directory='.',
                                 script_name='DataprepPipeline.py',
                                 inputs=[input_ds.as_named_input('raw_data')],
                                 outputs=[dataFolder],
                                 runconfig=run_config,
                                 arguments=['--datafolder', dataFolder])

# Step 02 - Train the model
train_step = PythonScriptStep(name='02 Train the Model',
                                 source_directory='.',
                                 script_name='TrainingPipeline.py',
                                 inputs=[dataFolder],
                                 runconfig=run_config,
                                 arguments=['--datafolder', dataFolder])








# Configure and build the pipeline
steps = [dataPrep_step, train_step]

from azureml.pipeline.core import Pipeline
new_pipeline = Pipeline(workspace=ws, steps=steps)


# Create the experiment and run the pipeline
from azureml.core import Experiment

new_experiment = Experiment(workspace=ws, name='PipelineExp01')
new_pipeline_run = new_experiment.submit(new_pipeline)

new_pipeline_run.wait_for_completion(show_output=True)






















