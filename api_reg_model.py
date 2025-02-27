from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib

# Create instance of FastAPI
app = FastAPI()

# Create a class for the request body
class request_body(BaseModel):
  study_hours : float
  
# Load the model
grade_model = joblib.load('./reg_model.pkl')

@app.post('/predict')
def predict(data : request_body):
  # Prepare data for prediction
  input_feature = [[data.study_hours]]
  
  # Make prediction
  y_pred = grade_model.predict(input_feature)[0].astype(int)
  
  return {'test_grade': y_pred.tolist()}