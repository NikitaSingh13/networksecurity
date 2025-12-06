# Network Security Project

This project provides a comprehensive solution for network security analysis, focusing on phishing detection and data-driven threat modeling. It leverages machine learning to process network data, validate inputs, and generate predictions.

## Features

- Data ingestion, validation, and transformation pipelines
- Machine learning model training and batch prediction
- Phishing data analysis using `phisingData.csv`
- Modular codebase for easy extension and maintenance
- Docker support for containerized deployment
- Output predictions saved to `prediction_output/output.csv`

## Project Structure

- `networksecurity/` – Core package with components for data processing, modeling, and utilities
- `Network_Data/` – Contains raw network data
- `valid_data/` – Stores validated test data
- `final_model/` & `mlflow_model_artifacts/` – Model artifacts and experiment tracking
- `templates/` – HTML templates for result visualization
- `notebooks/` – Jupyter notebooks for exploration

## Getting Started

1. Install dependencies:
	```
	pip install -r requirements.txt
	```
2. Run the main application:
	```
	python app.py
    ```

## Usage

- Customize data pipelines in `networksecurity/components/`
- Update model logic in `networksecurity/components/model_trainer.py`
- View prediction results in `prediction_output/output.csv`
