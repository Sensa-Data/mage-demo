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
    measurements_tag_columns = {
        "WaterQuality": ['Equipment', 'Section', 'Subunit', 'Unit', 'Origin'],
        "feedingsystem": ['Equipment', 'Section', 'Subunit', 'Unit', 'Origin'],
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
        df, data_quality_df, tags, *_ = measurement
        measurement_name = measurements[idx]
        measurement_columns = measurements_field_columns.get(measurement_name)
        tag_columns = measurements_tag_columns.get(measurement_name)
        anomaly_columns = measurements_anomaly_field_columns.get(measurement_name)

        model_path = measurements_trained_model_locations.get(measurement_name)
        trained_model = joblib.load(model_path)

        if trained_model is not None:
            ANOMALY_COL = anomaly_columns[0]  # for now, we only consider univariate models
            anomaly_df_cols = [f"{col_name}_Anomaly" for col_name in measurement_columns]
            df[anomaly_df_cols] = DEFAULT_ANOMALY_VALUE
            
            # batch
            df[TIME_COL] = pd.to_datetime(df[TIME_COL])
            filtered_df = df.filter([TIME_COL, ANOMALY_COL])
            # filtered_df.dropna(subset=[ANOMALY_COL], inplace=True)
            filtered_df['hour'] = filtered_df[TIME_COL].dt.hour
            filtered_df['day_of_week'] = filtered_df[TIME_COL].dt.dayofweek
            filtered_df.set_index("time", inplace=True)

            # TODO Remove fitting the scaler with testing data to avoid data leakage
            # save scaler in the .pkl file as well
            scaler = StandardScaler()
            filtered_df_scaled = scaler.fit_transform(filtered_df)

            anomaly_labels = trained_model.predict(filtered_df_scaled)
            anomalies = anomaly_labels == -1
            anomaly_scores = trained_model.decision_function(filtered_df_scaled)

            df[f"{ANOMALY_COL}_Anomaly"] = anomaly_scores.round(2)

            # TODO when multivariate models are ready, incorporate other anomaly scores as well
            source_groups = df.groupby(tag_columns)[f"{ANOMALY_COL}_Anomaly"].first().to_dict()
            data_quality_df["Aggregated_Anomaly"] = data_quality_df[tag_columns].apply(tuple, axis=1).map(source_groups)

            aggregated_data = aggregated_data + ((df, data_quality_df, tags),)
        
            # # single row
            # aggregated_anomaly_rows = []
            # filtered_df = df.filter([TIME_COL, OXYGEN_COL])
            # for index, anomaly_row in filtered_df.iterrows():
            #     anomaly_row_df = pd.DataFrame([anomaly_row])
            #     if pd.isna(anomaly_row_df[OXYGEN_COL]):
            #         continue

            #     anomaly_row_df[TIME_COL] = pd.to_datetime(anomaly_row_df[TIME_COL])
            #     anomaly_row_df['hour'] = anomaly_row_df[TIME_COL].dt.hour
            #     anomaly_row_df['day_of_week'] = anomaly_row_df[TIME_COL].dt.dayofweek
            #     anomaly_row_df.set_index("time", inplace=True)

            #     # TODO Remove fitting the scaler with testing data to avoid data leakage
            #     # save scaler in the .pkl file as well
            #     scaler = StandardScaler()
            #     row_df_scaled = scaler.fit_transform(anomaly_row_df)

            #     anomaly_labels = trained_model.predict(row_df_scaled)
            #     anomalies = anomaly_labels == -1
            #     anomaly_scores = trained_model.decision_function(row_df_scaled)

            #     th_filtered_anomaly_scores = anomaly_scores[anomaly_scores > ANOMALY_THRESHOLD]
            #     anomaly_row[f"{OXYGEN_COL}_Anomaly"] = th_filtered_anomaly_scores.mean().round(2)

            #     aggregated_anomaly_rows.append(anomaly_row)

            # aggregated_anomaly_df = pd.DataFrame(aggregated_anomaly_rows)
            # aggregated_data = aggregated_data + ((aggregated_anomaly_df, tags),)


    return aggregated_data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    # assert output is not None, 'The output is undefined'
    assert output is not None, 'The output is undefined'
