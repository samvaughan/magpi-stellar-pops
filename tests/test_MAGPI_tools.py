import pytest
from unittest.mock import patch
import pandas as pd
from src.scripts import MAGPI_tools
from astropy.io import fits


def mock_MAGPI_catalogue():
    df = pd.DataFrame(data=dict(MAGPIID=[1201027109, 1201029155], z=[0.001, 0.002]))
    return df


def mock_config_file():
    return dict(catalogue_filename="catalogue_location.fits")


def mock_cube_hdu():
    hdu1 = fits.PrimaryHDU()
    hdu2 = fits.ImageHDU(name="DATA")
    hdu3 = fits.ImageHDU(name="LSF")

    # Add the necessary DATA header keywords
    # This is all the WCS info stuff
    hdu2.header["CTYPE1"] = "RA---TAN"
    hdu2.header["CTYPE2"] = "DEC--TAN"
    hdu2.header["CTYPE3"] = "WAVE"

    hdu2.header["CRPIX1"] = 0
    hdu2.header["CRPIX2"] = 0
    hdu2.header["CRPIX3"] = 1.0

    hdu2.header["CRVAL1"] = 222.1465147830423
    hdu2.header["CRVAL2"] = 2.938900767171168
    hdu2.header["CRVAL3"] = 4700.0

    hdu2.header["CD1_1"] = -5.55555555555556e-05
    hdu2.header["CD1_2"] = 0.0
    hdu2.header["CD2_1"] = 0.0
    hdu2.header["CD2_2"] = 5.55555555555556e-05
    hdu2.header["CD3_3"] = 1.25
    hdu2.header["CD1_3"] = 0.0
    hdu2.header["CD2_3"] = 0.0
    hdu2.header["CD3_1"] = 0.0
    hdu2.header["CD3_2"] = 0.0

    hdu2.header["NAXIS"] = 3
    hdu2.header["NAXIS1"] = 94
    hdu2.header["NAXIS2"] = 84
    hdu2.header["NAXIS3"] = 3722

    # Add the necessary LSF header words
    hdu3.header["MAGPI LSF WMIN"] = 0
    hdu3.header["MAGPI LSF WMAX"] = 1
    hdu3.header["MAGPI LSF COEFF0"] = 0
    hdu3.header["MAGPI LSF COEFF1"] = 0
    hdu3.header["MAGPI LSF COEFF2"] = 0
    new_hdul = fits.HDUList([hdu1, hdu2, hdu3])

    return new_hdul


@pytest.fixture
def chosen_ID():
    return 1201027109


@pytest.fixture
def chosen_filename():
    return "test.fits"


@pytest.fixture
@patch("src.scripts.MAGPI_tools.fits.open")
@patch("src.scripts.MAGPI_tools.yaml.safe_load")
@patch("src.scripts.MAGPI_tools.pd.read_csv")
def magpi_spec_class(read_csv, safe_load, fits_open, chosen_ID, chosen_filename):
    # Get a very small table with just 2 IDs for our master catalogue table
    read_csv.return_value = mock_MAGPI_catalogue()
    # Mock loading the yaml config file too
    safe_load.return_value = mock_config_file()

    # And our cube
    fits_open.return_value = mock_cube_hdu()

    # chosen_ID = 1201027109
    # chosen_filename = "test.fits"
    m = MAGPI_tools.MAGPI_spec(filename=chosen_filename, ID=chosen_ID)
    return m


# @pytest.mark.parametrize("chosen_ID, chosen_filename", [(1201027109, "test.fits")])
def test_MAGPI_Spec_attributes(magpi_spec_class, chosen_ID, chosen_filename):
    assert magpi_spec_class.MAGPI_ID == chosen_ID
    assert magpi_spec_class.filename == chosen_filename
    assert type(magpi_spec_class.hdu) is fits.hdu.hdulist.HDUList


def test_MAGPI_Spec_class_has_redshift_info(
    magpi_spec_class,
):
    assert hasattr(magpi_spec_class, "redshift")


def test_MAGPI_Spec_class_has_wavelength_info(
    magpi_spec_class,
):
    assert hasattr(magpi_spec_class, "wavelength")


@pytest.mark.skip(reason="not implemented yet")
def test_MAGPI_Cube_class_has_correct_spec_types():
    assert False


@pytest.mark.skip(reason="not implemented yet")
def test_MAGPI_Cube_class_has_correct_variance_types():
    assert False
