import yaml
import pandas as pd
from astropy.io import fits
import numpy as np
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from astropy.visualization import LogStretch, ImageNormalize
from . import utils


class MAGPI_spec:
    """
    A general class for MAGPI spectra which other classes can inherit from
    """

    def __init__(self, filename, ID, config_file="MAGPI_config.yml"):
        with open(config_file, "r") as f:
            data = yaml.safe_load(f)

        self.MAGPI_ID = ID

        self.filename = filename
        self.catalogue_filename = data["catalogue_filename"]

        master_catalogue = pd.read_csv(self.catalogue_filename)
        assert (
            int(ID) in master_catalogue.MAGPIID.values
        ), f"Our ID of {ID} doesn't seem to be in the MAGPI master catalogue"

        # Get the data associated with our galaxy
        self.row = master_catalogue.loc[master_catalogue.MAGPIID == ID]

        # Open the file
        self.hdu = fits.open("{}".format(self.filename))

        # Add some useful properties for extensions of the fits files
        self.pri_header = self.hdu["PRIMARY"].header
        self.spec_header = self.hdu["DATA"].header
        self.LSF_header = self.hdu["LSF"].header

        # Check how many dimensions we have
        self.naxis = self.spec_header["NAXIS"]
        assert (self.naxis == 1) or (
            self.naxis == 3
        ), "NAXIS must either be 1 for a spectrum or 3 for a cube!"

        self.LSF_wave_range = np.array(
            [self.LSF_header["MAGPI LSF WMIN"], self.LSF_header["MAGPI LSF WMAX"]]
        )
        self.LSF_coefficients = np.array(
            [
                self.LSF_header["MAGPI LSF COEFF0"],
                self.LSF_header["MAGPI LSF COEFF1"],
                self.LSF_header["MAGPI LSF COEFF2"],
            ]
        )

        self.wcs = WCS(self.spec_header)
        self.redshift = self.get_redshift()
        self.wavelength = self.get_wavelength_calibration()

    def get_redshift(self):
        return self.row["z"]

    def get_wavelength_calibration(self):
        # Get the WCS info from the header

        if self.naxis == 3:
            n_wavelength_pixels = self.spec_header["NAXIS3"]
            _, _, wave = self.wcs.pixel_to_world_values(
                0, 0, np.arange(n_wavelength_pixels)
            )
        else:
            n_wavelength_pixels = self.spec_header["NAXIS1"]
            wave = self.wcs.pixel_to_world_values(np.arange(n_wavelength_pixels))

        return wave


class MAGPI_cube(MAGPI_spec):
    """
    A class to work with MAGPI data cubes
    """

    def __init__(self, cube_filename, ID, config_file):
        # Call init of MAGPI_spec
        super().__init__(filename=cube_filename, ID=ID, config_file=config_file)

        # Raw datacube
        self.unbinned_data = self.hdu[1].data
        # Variance
        self.unbinned_variance = self.hdu[2].data

        self.MAGPI_field = self.row["field"]
        self.image_wcs = self.wcs.celestial

        self.RA_pix, self.DEC_pix = utils.get_RA_DEC_values_of_pixels(
            self.spec_header, self.image_wcs
        )

        # Try and load the Voronoi binned spectra if they exist for this cube
        try:
            self.vor20_spec_collection = utils.get_spectrum1D_from_specs(
                wavelength=self.lamdas,
                flux_spectrum=self.hdu["VORBIN_SN_20"].data,
                variance_spectrum=self.hdu["VORBIN_SN_20_NOISE"].data,
                uncertainty_is_variance=False,
                flux_units=self.units,
                flux_factor=1,
                FWHM_gal=2.66,
            )
            self.vor20_quick_SN = super().quick_SN(
                self.vor20_spec_collection.spectrum.flux,
                self.vor20_spec_collection.spectrum.uncertainty,
            )
            self.vor20_mask = self.hdu["VORBIN_SN_20_BINMASK"].data
        except KeyError:
            pass

    def plot_whitelight_image(self, **kwargs):
        """
        Plot a white-light image of the cube by median-combining along the wavelength axis.
        Kwargs are passed to imshow
        """
        img = self._get_img()
        norm = ImageNormalize(vmin=img.min(), vmax=img.max(), stretch=LogStretch())

        fig, ax = plt.subplots(subplot_kw=dict(projection=self.image_wcs))
        ax.imshow(img, norm=norm, **kwargs)
        return fig, ax

    def _get_img(self):
        return np.nanmedian(self.unbinned_data, axis=0)
