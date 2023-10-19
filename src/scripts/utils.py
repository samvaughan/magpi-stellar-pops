from astropy.table import Table
from astropy.io import fits


def load_FITS_table_in_pandas(filename):
    """Load a fits table into a pandas table

    Args:
        filename (string): A fits table file

    Returns:
        pd.DataFrame: a pandas dataframe
    """
    hdu = fits.open(filename)
    table = Table(hdu[1].data)

    return table.to_pandas()
