from azureml.core import Workspace
from azureml.core import Environment
from azureml.core.environment import  CondaDependencies
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.steps import  PythonScriptStep
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import  Pipeline
from azureml.core import Experiment


"""
Access the workspace
"""
ws = Workspace.from_config("./config")
"""
###############################################################################
"""


"""
Create custom envirnmont
"""
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
"""
###############################################################################
"""

"""
create compute cluster for pipeline
"""
# give name of the cluster
cluster_name = "clusteralpha"

# set the configuration of th cluster
compute_config = AmlCompute.provisioning_configuration(
                                                        vm_size='STANDARD_D11_V2',
                                                        max_nodes=2
                                                      )
# biuld comute cluster
compute_cluster = ComputeTarget.create(ws, cluster_name, compute_config)

# wait for completion
compute_cluster.wait_for_completion()
"""
###############################################################################
"""


"""
Create Run Configuration for the steps 
"""
run_config = RunConfiguration()
run_config.target = compute_cluster
run_config.environment = pipeline_env

"""
###############################################################################
"""


"""
Pipeline steps 
"""
input_ds = ws.datasets.get("Defaults")
datafolder = PipelineData("datafolder", datastore=ws.get_default_datastore())

# Step 1 : Data peprocessing :
dataPreprocess_step = PythonScriptStep(
                                        name="01 Data preprocessing",
                                        source_directory=".",
                                        script_name="preprocessing_data_step.py",
                                        inputs=[input_ds.as_named_input("raw_data")],
                                        outputs=[datafolder],
                                        runconfig=run_config,
                                        arguments=["--datafolder", datafolder]
                                      )

# Configure and build the pipeline
steps = [dataPreprocess_step]
new_pipeline = Pipeline(
                        workspace=ws,
                        steps=steps
                       )

# create the experiment and run pipeline
new_experiment = Experiment(
                            workspace=ws,
                            name="PipelineExp01"
                           )
new_pipeline_run = new_experiment.submit(new_pipeline)
new_pipeline_run.wait_for_completion(show_output=True)
"""
###############################################################################
"""
