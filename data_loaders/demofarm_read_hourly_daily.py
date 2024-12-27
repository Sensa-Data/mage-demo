if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
from mage_ai.data_preparation.shared.secrets import get_secret_value
from datetime import datetime, timedelta
from influxdb_client_3 import InfluxDBClient3
import pandas as pd

#Reading and writing from same bucket
host = get_secret_value('influx_host')
org = get_secret_value('dev_org')
token = get_secret_value('demofarm_long_read_token')
bucket = 'demofarm_long'
measurements=["WaterQuality", "feedingsystem"]


@data_loader
def load_data(*args, **kwargs):
    """
    Read minute data from salmonique long

    Returns:
        tuple(minute data, tags)
    """
    
    schedule_time = kwargs.get('execution_date')
    stop_time = datetime.fromisoformat(str(schedule_time)).replace(tzinfo=None)
    trigger_interval = kwargs.get('trigger_interval' , None)
    # trigger_interval = 'day'
    if trigger_interval is None or trigger_interval.startswith('var'):
        raise Exception("Set trigger Interval")
    if trigger_interval =='hour':
        start_time = stop_time - timedelta(hours=1)
    elif trigger_interval =='day':
        start_time = stop_time - timedelta(days=1)
    else:
        raise Exception("trigger_interval can be hour or day only")

    minute_data = ()
    for measurement in measurements:
        with InfluxDBClient3(
                host=host,
                org=org,
                token=token,
                database=bucket
            ) as client:
                
                if trigger_interval =='hour':
                    data_frame_measurement_name = f"biofish_{measurement}_min"
                elif trigger_interval =='day':
                    data_frame_measurement_name = f"biofish_{measurement}_hour"
                else:
                    raise Exception("trigger_interval can be hour or day only")
                query = f'''
                    SELECT * 
                    FROM "{data_frame_measurement_name}"
                    WHERE
                    time >= '{start_time}'
                    AND time <= '{stop_time}'
                '''
                query_str_singleline = ' '.join(line.strip() for line in query.splitlines())
                print(f"Query on salmonique_long: \"{query_str_singleline.strip()}\"")
                table = client.query(query, language="influxql")
                table = table.drop("iox::measurement")

                if table.num_rows < 1:
                    print(f"Query: \"{query_str_singleline.strip()}\" returned empty table.")
                    raise Exception(f"Query: \"{query_str_singleline.strip()}\" returned empty table.")

                # Get list of tags
                tags = []
                for column in table.schema:
                    if column.metadata[b"iox::column::type"].endswith(b"tag"):
                        tags.append(column.name)
                df = table.to_pandas()
                minute_data = minute_data + ((df, tags),)

    return minute_data
@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'