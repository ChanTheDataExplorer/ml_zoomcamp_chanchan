import bentoml

from bentoml.io import JSON
from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str
    age: int
    country: str
    rating: float

model_ref = bentoml.xgboost.get("credit_risk_model:rswvfisptsphqtos")
dv = model_ref.custom_objects["dictVectorizer"]

model_runner = model_ref.to_runner()

svc = bentoml.Service("credit_risk_classifier", runners = [model_runner])

@svc.api(input=JSON(pydantic_model=UserProfile), output=JSON())
def classify(application_data):
    vector = dv.transform(application_data)
    prediction = model_runner.predict.run(vector)
    print(prediction)
    
    result = prediction[0]

    if result > 0.5:
        return {"status": "DECLINED"}
    elif result > 0.25:
        return {"status": "MAYBE"}
    else:
        return {"status": "APPROVED"}
    

