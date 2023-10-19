import yaml
import pandas as pd
from astropy.io import fits
import numpy as np


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

        # Open the minicube
        self.hdu = fits.open("{}".format(self.filename))

        # Add some useful properties for extensions of the fits files
        self.pri_header = self.hdu["PRIMARY"].header
        self.spec_header = self.hdu["DATA"].header
        self.LSF_header = self.hdu["LSF"].header

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
