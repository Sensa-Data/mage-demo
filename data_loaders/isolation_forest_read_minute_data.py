if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
from mage_ai.data_preparation.shared.secrets import get_secret_value

from datetime import datetime, timedelta
from influxdb_client_3 import InfluxDBClient3
import pandas as pd
import numpy as np

host = get_secret_value('influx_host')
org = get_secret_value('prod_org')
token = get_secret_value('demofarm_raw_read_token')

measurements=["WaterQuality", "feedingsystem"]
source_bucket = 'biofish_raw'


@data_loader
def load_data(*args, **kwargs):
    """
    Read raw data from biofish_min

    Returns:
        tuple(dataframe, tags)
    """
    schedule_time = kwargs.get('execution_date')
    stop_time = datetime.fromisoformat(str(schedule_time)).replace(tzinfo=None)
    start_time = stop_time - timedelta(minutes=1)

    dummy_output = []
    rawdata = ()
    for measurement in measurements:
        with InfluxDBClient3(
            host=host,
            org=org,
            token=token,
            database=source_bucket
        ) as client:
            query = f'''
                SELECT *
                FROM "{measurement}"
                WHERE
                time >= '{start_time}'
                AND time <= '{stop_time}'
            '''
            query_str_singleline = ' '.join(line.strip() for line in query.splitlines())
            print(f"Query on {source_bucket}: \"{query_str_singleline.strip()}\"")
            table = client.query(query, language="influxql")
            table = table.drop_columns(["iox::measurement", "Quality", "host", "id"])
            
            # These columns are empty in the 'feedingsystem' measurement (starting from 08/11-2023 and going forwards)
            if measurement == "feedingsystem":
                table = table.drop_columns(["Sensor", "Tank"])

            if table.num_rows < 1:
                print(f"Query: \"{query_str_singleline.strip()}\" returned empty table.")
                raise Exception(f"Query: \"{query_str_singleline.strip()}\" returned empty table.")

            # Get list of tags
            measurement_tags = []
            for column in table.schema:
                if column.metadata[b"iox::column::type"].endswith(b"tag"):
                    measurement_tags.append(column.name)
            measurement_tags.append("Origin")

            df = table.to_pandas()
            df["Origin"] = measurement
            rawdata = rawdata + ((df, measurement_tags),)

    return rawdata


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
