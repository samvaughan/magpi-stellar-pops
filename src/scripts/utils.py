from astropy.table import Table
from astropy.io import fits
import numpy as np
import astropy.units as u

from . import spec_tools as S


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


def get_RA_DEC_values_of_pixels(spec_header, wcs):
    """Given a fits header and a WCS object, return the RA and DEC of the pixels

    Args:
        spec_header (header): A FITS header
        wcs (WCS): The WCS information asociated with the header. Must be at least 2D

    Returns:
        tuple: A tuple of 2D arrays of RA and DEC coordinates for each pixel.
    """
    x = np.arange(spec_header["NAXIS1"])
    y = np.arange(spec_header["NAXIS2"])

    rr = wcs.celestial.pixel_to_world(x, y[0])
    ra_values = np.array([rr[i].ra.value for i in range(len(rr))])
    dd = wcs.celestial.pixel_to_world(x[0], y)
    dec_values = np.array([dd[i].dec.value for i in range(len(dd))])

    ra, dec = np.meshgrid(ra_values, dec_values)

    return ra, dec


def get_spectrum1D_from_specs(
    wavelength,
    flux_spectrum,
    variance_spectrum,
    flux_units=u.Unit("erg cm-2 s-1 AA-1"),
    wavelength_units="AA",
    flux_factor=1e-16,
    meta=None,
    uncertainty_is_variance=True,
    logrebinned=False,
    normalise=True,
    FWHM_gal=2.66,
    quiet=False,
):
    """
    Create a spectrum1D object from individual 1D spectra.
    Note that to not break the looping through spectra in the spectral fitting code,
    we make the flux and variance spectra two dimensional,
    with the first array full of nans and the second the actual (with shape (1, n))

    Args:
        wavelength (_type_): _description_
        flux_spectrum (_type_): _description_
        variance_spectrum (_type_): _description_
        flux_units (_type_, optional): _description_. Defaults to u.Unit("erg cm-2 s-1 AA-1").
        wavelength_units (str, optional): _description_. Defaults to "AA".
        flux_factor (_type_, optional): _description_. Defaults to 1e-16.
        meta (_type_, optional): _description_. Defaults to None.
        uncertainty_is_variance (bool, optional): _description_. Defaults to True.
        logrebinned (bool, optional): _description_. Defaults to False.
        normalise (bool, optional): _description_. Defaults to True.
        FWHM_gal (float, optional): _description_. Defaults to 2.66.
        quiet (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    spec_collection = S.SpectrumCollection(
        wavelength=wavelength.value,
        flux=flux_spectrum * flux_factor,
        uncertainty=variance_spectrum * flux_factor * flux_factor,
        wavelength_units=wavelength_units,
        flux_units=flux_units,
        uncertainty_is_variance=uncertainty_is_variance,
        meta=meta,
        logrebinned=logrebinned,
        normalise=normalise,
        FWHM_gal=FWHM_gal,
        quiet=quiet,
    )

    # spectrum=Spectrum1D(spectral_axis=wavelength, flux=np.atleast_2d(flux_spectrum)*flux_factor*unit, uncertainty=VarianceUncertainty(np.atleast_2d(variance_spectrum)*flux_factor*flux_factor), meta=meta)

    return spec_collection
