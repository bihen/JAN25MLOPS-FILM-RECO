import evidently
import datetime
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import json

from sklearn import datasets, ensemble, model_selection
from scipy.stats import anderson_ksamp

from evidently.metrics import RegressionQualityMetric, RegressionErrorPlot, RegressionErrorDistribution
from evidently.metric_preset import DataDriftPreset, RegressionPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.ui.workspace import Workspace

# ignore warnings
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# define features and target
TARGET = 'rating'
PREDICTION = 'predicted_score'

# define project
WORKSPACE_NAME = "datascientest-workspace"
PROJECT_NAME = "movie_recommender"
PROJECT_DESCRIPTION = "Evidently for JAN25 MLOPS Movie Recommender project"

# custom functions
def _fetch_data() -> pd.DataFrame:
    content = requests.get("https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip", verify=False).content
    with zipfile.ZipFile(io.BytesIO(content)) as arc:
        raw_data = pd.read_csv(arc.open("hour.csv"), header=0, sep=',', parse_dates=['dteday']) 
    return raw_data

def _process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    raw_data.index = raw_data.apply(lambda row: datetime.datetime.combine(row.dteday.date(), datetime.time(row.hr)),
                                    axis=1)
    return raw_data

def train_model(data):
    # Train test split
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES],
        data[TARGET],
        test_size=0.3
    )

    # Model training
    regressor = ensemble.RandomForestRegressor(random_state = 0, n_estimators = 50)
    regressor.fit(X_train, y_train)

    # Predictions
    preds_train = regressor.predict(X_train)
    X_train['target'] = y_train
    X_train['prediction'] = preds_train

    # Add actual target and prediction columns to the test data for later performance analysis
    preds_test = regressor.predict(X_test)
    X_test['target'] = y_test
    X_test['prediction'] = preds_test
    return regressor, X_train, X_test


def performance_report(data_ref, data_curr):
    # Initialize the column mapping object which evidently uses to know how the data is structured
    column_mapping = ColumnMapping()

    # Map the actual TARGET and prediction column names in the dataset for evidently
    column_mapping.target = 'target'
    column_mapping.prediction = 'prediction'

    # Specify which features are numerical and which are categorical for the evidently report
    column_mapping.numerical_features = NUMERICAL_FEATURES
    column_mapping.categorical_features = CATEGORICAL_FEATURES

    # Initialize the regression performance report with the default regression metrics preset
    regression_performance_report = Report(metrics=[
        RegressionPreset(),
    ])
    # Run the regression performance report using the training data as reference and test data as current
    # The data is sorted by index to ensure consistent ordering for the comparison
    if data_ref.empty:     
        regression_performance_report.run(reference_data=None, 
                        current_data=data_curr.sort_index(),
                        column_mapping=column_mapping)
    else:
        regression_performance_report.run(reference_data=data_ref.sort_index(), 
                                current_data=data_curr.sort_index(),
                                column_mapping=column_mapping)
    return regression_performance_report


def production_drift(regressor, ref_data):
    # Train the production model
    regressor.fit(ref_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES], ref_data[TARGET])

    # Perform column mapping
    column_mapping = ColumnMapping()
    column_mapping.target = TARGET
    column_mapping.prediction = PREDICTION
    column_mapping.numerical_features = NUMERICAL_FEATURES
    column_mapping.categorical_features = CATEGORICAL_FEATURES

    # Generate predictions for the reference data
    ref_prediction = regressor.predict(ref_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])
    ref_data_target = ref_data[TARGET]
    ref_data = ref_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    ref_data['target'] = ref_data_target
    ref_data['prediction'] = ref_prediction

    # Initialize the regression performance report with the default regression metrics preset
    regression_performance_report = Report(metrics=[
        RegressionPreset(),
    ])

    # Run the regression performance report using the reference data
    regression_performance_report.run(reference_data=None, 
                                    current_data=ref_data,
                                    column_mapping=column_mapping)
    return regressor, ref_data, regression_performance_report

def data_drift(ref_data, curr_data):
    column_mapping_drift = ColumnMapping()
    column_mapping_drift.target = TARGET
    column_mapping_drift.prediction = PREDICTION
    column_mapping_drift.numerical_features = NUMERICAL_FEATURES
    column_mapping_drift.categorical_features = []

    data_drift_report = Report(metrics=[
        DataDriftPreset(),
    ])

    data_drift_report.run(
        reference_data=ref_data,
        current_data=curr_data,
        column_mapping=column_mapping_drift,
    )
    return data_drift_report

def target_drift(ref_data, curr_data):
    column_mapping_drift = ColumnMapping()
    column_mapping_drift.target = TARGET
    column_mapping_drift.prediction = PREDICTION
    column_mapping_drift.numerical_features = NUMERICAL_FEATURES
    column_mapping_drift.categorical_features = CATEGORICAL_FEATURES

    target_drift_report = Report(metrics=[
        TargetDriftPreset(),
    ])

    target_drift_report.run(
        reference_data=ref_data,
        current_data=curr_data,
        column_mapping=column_mapping_drift,
    )
    return target_drift_report

def add_report_to_workspace(workspace, project_name, project_description, report, report_name, report_tags=[]):
    """
    Adds a report to an existing or new project in a workspace with a specific name and optional tags.
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
    report.tags = report_tags  
    workspace.add_report(project.id, report)
    
    print(f"New report '{report_name}' added to project {project_name} with tags: {report_tags}")
    
if __name__ == "__main__":
    # load data
    raw_data = _process_data(_fetch_data())
    
    #create workspace
    workspace = Workspace.create(WORKSPACE_NAME)
    
    # Reference and current data split
    ref_data = raw_data.loc['2011-01-01 00:00:00':'2011-01-28 23:00:00']
    week1 = raw_data.loc['2011-01-29 00:00:00' : '2011-02-07 23:00:00']
    week2 = raw_data.loc['2011-02-07 00:00:00' : '2011-02-14 23:00:00']
    week3 = raw_data.loc['2011-02-15 00:00:00' : '2011-02-21 23:00:00']
    
    # train model and create validation report
    regressor, X_train, X_test = train_model(ref_data)
    model_validation_report = performance_report(X_train, X_test)
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, model_validation_report, "Model Validation report", ["Validation"])
    
    # build production model on whole dataset
    production_regressor = regressor.fit(ref_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES], ref_data[TARGET])
    
    # predict target for production regressor
    ref_prediction = production_regressor.predict(ref_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])
    prod_data = ref_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    prod_data['target'] = ref_data[TARGET]
    prod_data['prediction'] = ref_prediction
    
    # Production model drift report
    production_report = performance_report(data_ref = pd.DataFrame(), data_curr = prod_data)
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, production_report, "Production model drift report", ["Production"])

    
    # create monitoring reports
    ## Week1
    week1_preds = production_regressor.predict(week1[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])
    week1_data = week1[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    week1_data["prediction"] = week1_preds
    week1_data["target"] = week1[TARGET]
    monitoring_report_week1 = performance_report(prod_data, week1_data)
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, monitoring_report_week1, "Week 1 drift report", ["Week 1"])
    
    ## Week2
    week2_preds = production_regressor.predict(week2[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])
    week2_data = week2[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    week2_data["prediction"] = week2_preds
    week2_data["target"] = week2[TARGET]
    monitoring_report_week2 = performance_report(prod_data, week2_data)
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, monitoring_report_week2, "Week 2 drift report", ["Week 2"])
    
    ## Week3
    week3_preds = production_regressor.predict(week3[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])
    week3_data = week3[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    week3_data["prediction"] = week3_preds
    week3_data["target"] = week3[TARGET]
    monitoring_report_week3 = performance_report(prod_data, week3_data)
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, monitoring_report_week3, "Week 3 drift report", ["Week 3"])

    # target drift report on the worst weeks
    target_drift_report_week2 = target_drift(prod_data, week2_data)
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, target_drift_report_week2, "Week 2 target drift report", ["Week 2", "Target drift"])
    
    target_drift_report_week3 = target_drift(prod_data, week3_data)
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, target_drift_report_week3, "Week 3 target drift report", ["Week 3", "Target drift"])
    
    # data drift report on the last week
    data_drift_report = data_drift(prod_data, week3_data)
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, data_drift_report, "Week 3 data drift report", ["Week 3", "Data drift"])