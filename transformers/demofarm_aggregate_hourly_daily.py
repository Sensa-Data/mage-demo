if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
import pandas as pd

@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        tuple(downsampled minute data, tags)
    """
    # Specify your transformation logic here
    aggregated_data = ()
    schedule_time = kwargs.get('execution_date')
    for measurement in data:
        df = measurement[0]
        tags = measurement[1]
        aggregated = df.groupby(tags).mean().reset_index()
        aggregated["time"] = schedule_time
        print(f"Downsampled {len(df.index)} rows into {len(aggregated.index)} rows.")
        aggregated_data = aggregated_data + ((aggregated, tags),)
    
    return aggregated_data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'