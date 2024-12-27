import pandas as pd
import numpy as np

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

    measurements=["WaterQuality", "feedingsystem"]
    measurements_field_columns = {
        "WaterQuality": ['Bisulfide', 'CO2', 'Conductivity', 'H2S', 'Nitrate', 'Nitrite', 'Oxygen', 'PH', 'TOCeq', 'Temperature', 'Turbidity', 'UV254f', 'UV254t'],
        "feedingsystem": ['AvgFeedHour'],
    }
    TIME_COL = 'time'
    MEASURING_UNIT_COL = 'MeasuringUnit'

    schedule_time = kwargs.get('execution_date')
    aggregated_data = ()

    for idx, measurement in enumerate(data):
        df = measurement[0]
        tags = measurement[1]
        measurement_name = measurements[idx]
        measurement_columns = measurements_field_columns.get(measurement_name)

        field_column_unit_aggregation = {}
        field_column_unit_aggregation[TIME_COL] = pd.to_datetime(df[TIME_COL]).mean()

        for field_column_name in measurement_columns:
            field_column_df = df[[TIME_COL, MEASURING_UNIT_COL, field_column_name]].copy()
            field_column_df = field_column_df.dropna(subset=[field_column_name])
            fc_unique_values = field_column_df[MEASURING_UNIT_COL].unique()

            assert len(fc_unique_values) == 1 # Later we will remove single MeasuringUnit assertion

            fc_non_null_mean = field_column_df[field_column_name].mean()
            field_column_unit_aggregation[field_column_name] = [fc_non_null_mean]
            field_column_unit_aggregation[f"{field_column_name}_Unit"] = fc_unique_values

        single_row_aggregated_df = pd.DataFrame(field_column_unit_aggregation)
        single_row_aggregated_df.head()

        
        aggregated_data = aggregated_data + ((single_row_aggregated_df, tags),)

    return aggregated_data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    # assert output is not None, 'The output is undefined'
    assert output is not None, 'The output is undefined'