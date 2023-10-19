import pytest
from src.scripts import utils
import pandas as pd

example_fits_table = "tests/test_data/example_fits_table.fits"


@pytest.mark.parametrize("fits_table_filename", [example_fits_table])
def test_loading_fits_table_gives_pandas_df(fits_table_filename):
    df = utils.load_FITS_table_in_pandas(fits_table_filename)
    assert type(df) is pd.pandas.core.frame.DataFrame
