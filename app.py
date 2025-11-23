# //we'll create apis using the fastAPI
import sys
import os

# ❌ NOT NEEDED FOR DEPLOYMENT
# import certifi
# ca = certifi.where()

# ❌ ENV + MONGO NOT NEEDED IN DEPLOYMENT
# from dotenv import load_dotenv
# load_dotenv()
# mongo_db_url = os.getenv("MONGODB_URL_KEY")
# print(mongo_db_url)

# ❌ pymongo not needed
# import pymongo

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

# ❌ training pipeline cannot run on Render
# from networksecurity.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

# ❌ MongoDB client not needed for predictions
# client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

# ❌ Constants for ingestion not needed for prediction
# from networksecurity.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
# from networksecurity.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME

# ❌ Not needed
# database = client[DATA_INGESTION_DATABASE_NAME]
# collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

# ❌ Training route removed for Render deployment
# @app.get("/train")
# async def train_route():
#     try:
#         train_pipeline = TrainingPipeline()
#         train_pipeline.run_pipeline()
#         return Response("Training is successful")
#     except Exception as e:
#         raise NetworkSecurityException(e, sys)

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        # Load saved artifacts
        preprocesor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")

        network_model = NetworkModel(preprocessor=preprocesor, model=final_model)

        y_pred = network_model.predict(df)

        df['predicted_column'] = y_pred

        df.to_csv('prediction_output/output.csv')

        table_html = df.to_html(classes='table table-striped')

        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)

# ❌ Remove run block — Docker/Render will run using uvicorn automatically
# if __name__ == "__main__":
#     app_run(app, host="0.0.0.0", port=8000)
