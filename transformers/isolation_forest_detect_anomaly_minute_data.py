import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from mage_ai.settings.repo import get_repo_path

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    """
    Downsample raw sensor data by minute

    Args:
        data: minute raw data and tags
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        tuple(dataframe, tags)
    """
    # Specify your transformation logic here

    project_path = get_repo_path()
    measurements=["WaterQuality", "feedingsystem"]
    measurements_field_columns = {
        "WaterQuality": ['Bisulfide', 'CO2', 'Conductivity', 'H2S', 'Nitrate', 'Nitrite', 'Oxygen', 'PH', 'TOCeq', 'Temperature', 'Turbidity', 'UV254f', 'UV254t'],
        "feedingsystem": ['AvgFeedHour'],
    }
    measurements_anomaly_field_columns = {
        "WaterQuality": ['Oxygen'],
        "feedingsystem": ['AvgFeedHour'],
    }


    measurements_trained_model_locations = {
        "WaterQuality": f"{project_path}/utils/models/isolation_forest/isolation_forest_model.pkl",
        "feedingsystem": f"{project_path}/utils/models/isolation_forest/isolation_forest_model_feeding.pkl",
    }

    TIME_COL = 'time'
    OXYGEN_COL = 'Oxygen'
    AVG_FEED_COL = 'AvgFeedHour'
    MEASURING_UNIT_COL = 'MeasuringUnit'

    ANOMALY_THRESHOLD = 0
    DEFAULT_ANOMALY_VALUE = 0

    schedule_time = kwargs.get('execution_date')
    aggregated_data = ()

    for idx, measurement in enumerate(data):
        df = measurement[0]
        tags = measurement[1]
        measurement_name = measurements[idx]
        measurement_columns = measurements_field_columns.get(measurement_name)

        model_path = measurements_trained_model_locations.get(measurement_name)
        trained_model = joblib.load(model_path)
        
        if measurement_name == "feedingsystem":
            aggregated_data = aggregated_data + ((df, tags),)

        elif measurement_name == "WaterQuality":

            for field_column_name in measurement_columns:
                if field_column_name != OXYGEN_COL:
                    df[f"{field_column_name}_Anomaly"] = [DEFAULT_ANOMALY_VALUE]
                    continue

                df[TIME_COL] = pd.to_datetime(df[TIME_COL])
                filtered_df = df.filter([TIME_COL, OXYGEN_COL])
                filtered_df.dropna(subset=[OXYGEN_COL], inplace=True)
                filtered_df['hour'] = filtered_df[TIME_COL].dt.hour
                filtered_df['day_of_week'] = filtered_df[TIME_COL].dt.dayofweek
                filtered_df.set_index("time", inplace=True)

                # DO NOT fit testing data on the scaler! data leakage
                # TODO save scaler in the .pkl file as well 
                scaler = StandardScaler()
                filtered_df_scaled = scaler.fit_transform(filtered_df)

                anomaly_labels = trained_model.predict(filtered_df_scaled)
                anomalies = anomaly_labels == -1
                anomaly_scores = trained_model.decision_function(filtered_df_scaled)

                th_filtered_anomaly_scores = anomaly_scores[anomaly_scores > ANOMALY_THRESHOLD]
                df[f"{OXYGEN_COL}_Anomaly"] = [th_filtered_anomaly_scores.mean()]


            aggregated_data = aggregated_data + ((df, tags),)

    return aggregated_data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    # assert output is not None, 'The output is undefined'
    assert output is not None, 'The output is undefined'
