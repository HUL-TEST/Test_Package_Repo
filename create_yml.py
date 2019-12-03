from azureml.core.conda_dependencies import CondaDependencies

myenv = CondaDependencies()
myenv.add_conda_package("scikit-learn")
myenv.add_conda_package("numpy")
myenv.add_conda_package("pandas")
myenv.add_pip_package("azureml-sdk[automl]")

env_file = "./test_env.yml"
with open(env_file,"w") as f:
    f.write(myenv.serialize_to_string())
print("Saved dependency info in", env_file)