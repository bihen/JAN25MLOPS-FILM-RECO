import evidently
import pandas as pd
import json
import os
import datetime
from pathlib import Path

from evidently.metrics import RegressionQualityMetric, RegressionErrorPlot, RegressionErrorDistribution
from evidently.metric_preset import DataDriftPreset, RegressionPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.ui.workspace import Workspace

import random
import joblib

# Define the paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FOLDER = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "data", "processed")
REFERENCE_FOLDER = os.path.join(BASE_DIR, "data", "reference_data")


# Define the project and workspace
WORKSPACE_NAME = "movie_recommender_workspace"
PROJECT_NAME = "movie_recommender_project"
PROJECT_DESCRIPTION = "Movie recommender model monitoring with Evidently"

# Define target and predicted columns
TARGET = 'rating'
PREDICTION = 'predicted_score'

# Define the user function to check for data drift
def load_data():
    """
    Load ratings and predictions data
    """
    ratings = pd.read_csv(os.path.join(RAW_FOLDER, "ratings.csv"))
    predictions = pd.read_csv(os.path.join(PROCESSED_FOLDER, "predictions.csv"))
    reference_ratings = pd.read_csv(os.path.join(REFERENCE_FOLDER, "ratings.csv"))
    reference_predictions = pd.read_csv(os.path.join(REFERENCE_FOLDER, "predictions.csv"))
    return ratings, predictions, reference_ratings, reference_predictions

def create_report(workspace, reference_data, current_data, column_mapping, report_name):
    """
    Creates a report based on data drift or target drift metrics.
    """
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(
        reference_data=reference_data.sort_index(),
        current_data=current_data.sort_index(),
        column_mapping=column_mapping
    )
    # Add the report to the workspace
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, report, report_name)
    return report
    
def add_report_to_workspace(workspace, project_name, project_description, report, report_name):
    """
    Adds a report to the workspace.
    """
    # Check if project already exists
    project = None
    for p in workspace.list_projects():
        if p.name == project_name:
            project = p
            break

    # Create a new project if it doesn't exist
    if project is None:
        project = workspace.create_project(project_name)
        project.description = project_description

    # Add report to the project
    report.metadata = {"title": report_name, "created_at": datetime.datetime.now().isoformat()}
    workspace.add_report(project.id, report)
    print(f"New report '{report_name}' added to project {project_name}")

def check_for_drift(workspace):
    """
    Check for data drift and prediction drift
    """
    ratings, predictions, reference_ratings, reference_predictions = load_data()

    # Ensure that we match ratings with predictions based on the user and movie
    merged_current_data = pd.merge(ratings[['userId', 'movieId', TARGET]], predictions[['user', 'movieId', PREDICTION]], 
                           left_on=['userId', 'movieId'], right_on=['user', 'movieId'], how='inner')
    
    # Ensure that we match ratings with predictions based on the user and movie
    merged_reference_data = pd.merge(reference_ratings[['userId', 'movieId', TARGET]], reference_predictions[['user', 'movieId', PREDICTION]], 
                           left_on=['userId', 'movieId'], right_on=['user', 'movieId'], how='inner')

    # Set up column mapping for Evidently
    column_mapping = ColumnMapping()
    column_mapping.target = TARGET
    column_mapping.prediction = PREDICTION
    column_mapping.numerical_features = ['userId', 'movieId']  # Assuming these are numerical features

    # Split data for drift analysis
    reference_data = merged_reference_data
    current_data = merged_current_data

    # Create the drift report
    drift_report = create_report(workspace, reference_data, current_data, column_mapping, "Data and Target Drift Report")
    return drift_report

if __name__ == "__main__":
    # Run the drift detection process
    workspace = Workspace.create(WORKSPACE_NAME)
    
    drift_report = check_for_drift(workspace)
    print("Data drift report generated and uploaded to the workspace.")