import bentoml

from bentoml.io import NumpyNdarray

bento_model = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5")

model_runner = bento_model.to_runner()

svc = bentoml.Service("mlzoomcamp_homework", runners = [model_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(application_data):
    prediction = model_runner.predict.run(application_data)
    print(prediction)

    return prediction
    

