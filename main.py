# Import the Workspace and Datastore
from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment
from azureml.core.environment import CondaDependencies



"""
Access the workspace, Datastore and Datasets
"""
ws = Workspace.from_config("./config")
"""
###############################################################################
"""

env_restored = ""
name_restored = ""

for name,env in ws.environments.items():
    if name == "pipeline_env":
        name_restored = name

        env_restored = env.version
    print('test', env.get_image_details(ws))
print("Name {} \t version {}".format(name,env.version))


restored_environment = Environment.get(workspace=ws,name=str(name_restored), version=str(env_restored))
print()
print("restored enviroment : ")
print("---------------------")
print(restored_environment)