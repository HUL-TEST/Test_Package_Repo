from azureml.core.image import Image,ContainerImage
from azureml.core import Workspace,Model

ws = Workspace.from_config()
model = ws.models['stack_regressor']

################# Method 1

from azureml.core.webservice import AciWebservice,Webservice
from azureml.core.model import InferenceConfig

inference_config = InferenceConfig(runtime= "python",
                                   source_directory = '.',
                                   entry_script="score.py",
                                   conda_file="test_env.yml")

deployment_config = AciWebservice.deploy_configuration()

from azureml.core.image import Image,ContainerImage
image_config = ContainerImage.image_configuration(runtime = 'python',
                                                 execution_script='score.py',
                                                 conda_file='test_env.yml',
                                                tags = {'type' : 'regression'},
                                                 description='ElasticNet Stacking Regression example'
                                                 )

image = Image.create(name = 'test-image-stack',
                    models = [model],
                    image_config=image_config,
                    workspace=ws)

image.wait_for_creation(show_output=True)

service_name = "service1"
try:
    Webservice(ws, service_name).delete()
except WebserviceException:
    pass

service = Model.deploy(ws, service_name, [model], inference_config)
service.wait_for_deployment(True)
print(service.state)


######################### Method 2

from azureml.core.webservice import Webservice, AciWebservice
from azureml.core.image import ContainerImage

aciconfig = AciWebservice.deploy_configuration(
                      tags={"data": "tabular",  "method" : "stacking"}, 
                      description='ElasticNet Stacking Regression')

image_config = ContainerImage.image_configuration(execution_script="score.py", 
                      runtime="python", 
                      conda_file="test_env.yml")



I am using Azure Machine Learning Service to deploy a ML model as web service.

I registered a model and now would like to deploy it as an ACI web service as in the guide.

To do so I define

from azureml.core.webservice import Webservice, AciWebservice
from azureml.core.image import ContainerImage

aciconfig = AciWebservice.deploy_configuration(cpu_cores=4, 
                      memory_gb=32, 
                      tags={"data": "text",  "method" : "NB"}, 
                      description='Predict something')

and

image_config = ContainerImage.image_configuration(execution_script="score.py", 
                      docker_file="Dockerfile",
                      runtime="python", 
                      conda_file="myenv.yml")


image = ContainerImage.create(name = "stack-image",
                      models = [model],
                      image_config = image_config,
                      workspace = ws
                      )

image.wait_for_creation(show_output=True)

service_name = 'service1'
service = Webservice.deploy_from_image(deployment_config = aciconfig,
                                        image = image,
                                        name = service_name,
                                        workspace = ws)