import os
from azureml.core import Workspace
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import Experiment, RunConfiguration
from azureml.core import ScriptRunConfig
from azureml.core.compute import ComputeTarget

ws = Workspace.from_config()
exp = Experiment(workspace = ws,name = 'test_exp_1')

comp_target = ComputeTarget(workspace = ws,name = 'compute1')
comp_target.wait_for_completion(show_output=True)

# create a new RunConfig object
run_config = RunConfiguration(framework='python')
run_config.target = comp_target
run_config.environment.python.user_managed_dependencies = False
run_config.environment.python.conda_dependencies = CondaDependencies.create(conda_packages=['scikit-learn',"numpy","pandas"])


# Create a script config
src = ScriptRunConfig(source_directory='.', 
                      script='train.py',
                      run_config=run_config) 

# submit the experiment
run = exp.submit(config=src)
run.wait_for_completion(show_output=True)

run.get_metrics()
run.get_file_names()
run.register_model(model_path='./outputs/best_model_stack.pkl',
 model_name='stack_regressor', tags={'Training context':'Experiment script'}, 
 properties={'RMSE': run.get_metrics()['rmse_score'], 'RMSE': run.get_metrics()['rmse_score']})

model = ws.models['stack_regressor']
