import pytest
from unittest.mock import patch
import pandas as pd
from src.scripts import MAGPI_tools
from astropy.io import fits


def mock_MAGPI_catalogue():
    df = pd.DataFrame(columns=["MAGPIID"], data=[1201027109, 1201029155])
    return df


def mock_config_file():
    return dict(catalogue_filename="catalogue_location.fits")


def mock_cube_hdu():
    hdu1 = fits.PrimaryHDU()
    hdu2 = fits.ImageHDU(name="DATA")
    hdu3 = fits.ImageHDU(name="LSF")

    # Add the necessary LSF header words
    hdu3.header["MAGPI LSF WMIN"] = 0
    hdu3.header["MAGPI LSF WMAX"] = 1
    hdu3.header["MAGPI LSF COEFF0"] = 0
    hdu3.header["MAGPI LSF COEFF1"] = 0
    hdu3.header["MAGPI LSF COEFF2"] = 0
    new_hdul = fits.HDUList([hdu1, hdu2, hdu3])

    return new_hdul


@patch("src.scripts.MAGPI_tools.fits.open")
@patch("src.scripts.MAGPI_tools.yaml.safe_load")
@patch("src.scripts.MAGPI_tools.pd.read_csv")
def test_MAGPI_Spec_attributes(read_csv, safe_load, fits_open):
    # Get a very small table with just 2 IDs for our master catalogue table
    read_csv.return_value = mock_MAGPI_catalogue()
    # Mock loading the yaml config file too
    safe_load.return_value = mock_config_file()

    # And our cube
    fits_open.return_value = mock_cube_hdu()

    chosen_ID = 1201027109
    chosen_filename = "test.fits"
    m = MAGPI_tools.MAGPI_spec(filename=chosen_filename, ID=chosen_ID)

    assert m.MAGPI_ID == chosen_ID
    assert m.filename == chosen_filename
    assert type(m.hdu) is fits.hdu.hdulist.HDUList


@pytest.mark.skip(reason="not implemented yet")
def test_MAGPI_Cube_has_wavelength_attributes():
    assert False


@pytest.mark.skip(reason="not implemented yet")
def test_MAGPI_Cube_class_has_redshift_info():
    assert False
