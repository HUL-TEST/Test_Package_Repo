from azureml.core import Run
import azureml

def get_amlrun():
    try:
        run = Run.get_submitted_run()
        return run
    except Exception as e:
        print('Exception error: %s' % (e.message))
        return None