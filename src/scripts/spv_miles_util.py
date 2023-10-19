###############################################################################
#
# Copyright (C) 2016-2018, Michele Cappellari
# E-mail: michele.cappellari_at_physics.ox.ac.uk
#
# This software is provided as is without any warranty whatsoever.
# Permission to use, for non-commercial purposes is granted.
# Permission to modify for personal or internal use is granted,
# provided this copyright and disclaimer are included unchanged
# at the beginning of the file. All other rights are reserved.
#
###############################################################################

# This file contains the 'miles' class with functions to contruct a
# library of MILES templates and interpret and display the output
# of pPXF when using those templates as input.

# SPV has edited this file to include the three dimensions of metallicity, age and alpha/Fe

from os import path
import glob
import re

import numpy as np
from scipy import ndimage
from astropy.io import fits

from ppxf import ppxf_util as util

###############################################################################
# MODIFICATION HISTORY:
#   V1.0.0: Written. Michele Cappellari, Oxford, 27 November 2016
#   V1.0.1: Assume seven characters for the age field. MC, Oxford, 5 July 2017
#   V1.0.2: Robust string matching to deal with different MILES conventions.
#       MC, Oxford, 29 November 2017
#   Edited by Sam on 23/7/19 to include alpha enhanced templates


def age_metal_alpha(filename):
    """
    Extract the age and metallicity from the name of a file of
    the MILES library of Single Stellar Population models as
    downloaded from http://miles.iac.es/ as of 2017

    This function relies on the MILES file containing a substring of the
    precise form like Zm0.40T00.0794, specifying the metallicity and age.

    :param filename: string possibly including full path
        (e.g. 'miles_library/Mun1.30Zm0.40T00.0794.fits')
    :return: age (Gyr), [M/H]

    """
    s = re.findall(r"Z[m|p][0-9]\.[0-9]{2}T[0-9]{2}\.[0-9]{4}", filename)[0]
    metal = s[:6]
    age = float(s[7:])

    # Added by Sam
    # I'm assuming alpha enhancements can never be negative- which is correct as of July 2019
    try:
        r = re.findall(r"i[T|P][m|p][0-9]\.[0-9]{2}_E[m|p][0-9]\.[0-9]{2}", filename)[0]
        alpha = float(r.split("_")[-1][-3:])
    except (
        IndexError
    ):  # If we get an index error, it's because this our regex failed to find anything
        alpha = 0.0

    if "Zm" in metal:
        metal = -float(metal[2:])
    elif "Zp" in metal:
        metal = float(metal[2:])

    return age, metal, alpha


###############################################################################
# MODIFICATION HISTORY:
#   V1.0.0: Adapted from my procedure setup_spectral_library() in
#       ppxf_example_population_sdss(), to make it a stand-alone procedure.
#     - Read the characteristics of the spectra directly from the file names
#       without the need for the user to edit the procedure when changing the
#       set of models. Michele Cappellari, Oxford, 28 November 2016
#   V1.0.1: Check for files existence. MC, Oxford, 31 March 2017
#   V1.0.2: Included `max_age` optional keyword. MC, Oxford, 4 October 2017
#   V1.0.3: Included `metal` optional keyword. MC, Oxford, 29 November 2017
#   V1.0.4: Changed imports for pPXF as a package. MC, Oxford, 16 April 2018
#   V1.1.0: Replaced ``normalize`, `max_age`, `min_age` and 'metal` keywords
#       with `norm_range`, `age_range` and `metal_range`.
#       MC, Oxford, 23 November 2018
#   Changed by Sam to include alpha enhancements July 2019


class miles(object):
    def __init__(
        self,
        pathname,
        velscale,
        FWHM_gal,
        FWHM_tem=2.51,
        age_range=None,
        metal_range=None,
        norm_range=None,
    ):
        """
        Produces an array of logarithmically-binned templates by reading
        the spectra from the Single Stellar Population (SSP) library by
        Vazdekis et al. (2010, MNRAS, 404, 1639) http://miles.iac.es/.
        The code checks that the model specctra form a rectangular grid
        in age and metallicity and properly sorts them in both parameters.
        The code also returns the age and metallicity of each template
        by reading these parameters directly from the file names.
        The templates are broadened by a Gaussian with dispersion
        sigma_diff = np.sqrt(sigma_gal**2 - sigma_tem**2).

        Thie script relies on the files naming convention adopted by
        the MILES library, where SSP spectra have the form like below

            *Zm0.40T00.0794*.fits
            (e.g. Mun1.30Zm0.40T00.0794_iPp0.00_baseFe_linear_FWHM_2.51.fits)

        This code can be easily adapted by the users to deal with other stellar
        libraries, different IMFs or different abundances.

        :param pathname: path with wildcards returning the list files to use
            (e.g. 'miles_models/Mun1.30*.fits'). The files must form a Cartesian
            grid in age and metallicity and the procedure returns an error if
            they do not.
        :param velscale: desired velocity scale for the output templates
            library in km/s (e.g. 60). This is generally the same or an integer
            fraction of the velscale of the galaxy spectrum.
        :param FWHM_gal: scalar or vector with the FWHM of the instrumental
            resolution of the galaxy spectrum in Angstrom at every pixel of
            the stellar templates.
        :param age_range: [age_min, age_max] optional age range (inclusive) in
            Gyr for the MILES models. This can be useful e.g. to limit the
            templates age to be younger than the age of the Universe at a given
            redshift.
        :param metal_range: [metal_min, metal_max] optional metallicity [M/H]
            range (inclusive) for the MILES models (e.g. metal_range = [0, 10]
            to select only the spectra with Solar metallicity and above).
        :param norm_range: Two-elements vector specifying the wavelength range
            in Angstrom within which to compute the templates normalization
            (e.g. norm_range=[5070, 5950] for the FWHM of the V-band).
          - When this keyword is set, the templates are normalized to
            np.mean(template[band]) = 1 in the given wavelength range.
          - When this keyword is used, ppxf will output light weights, and
            mean_age_metal() will provide light-weighted stellar population
            quantities.
          - If norm_range=None (default), the templates are not normalized.
        :return: The following variables are attributes of the miles class:
            .templates: array has dimensions templates[npixels, n_ages, n_metals];
            .log_lam_temp: natural np.log() wavelength of every pixel npixels;
            .age_grid: (Gyr) has dimensions age_grid[n_ages, n_metals];
            .metal_grid: [M/H] has dimensions metal_grid[n_ages, n_metals].
            .n_ages: number of different ages
            .n_metal: number of different metallicities

        """

        self.pathname = pathname
        files = glob.glob(path.expanduser(pathname))
        assert len(files) > 0, "Files not found %s" % pathname

        all = [age_metal_alpha(f) for f in files]
        all_ages, all_metals, all_alphas = np.array(all).T
        ages, metals, alphas = (
            np.unique(all_ages),
            np.unique(all_metals),
            np.unique(all_alphas),
        )
        n_ages, n_metal, n_alpha = len(ages), len(metals), len(alphas)

        assert set(all) == set(
            [(a, b, c) for a in ages for b in metals for c in alphas]
        ), "Ages and Metals do not form a Cartesian grid"

        # Extract the wavelength range and logarithmically rebin one spectrum
        # to the same velocity scale of the galaxy spectrum, to determine the
        # size needed for the array which will contain the template spectra.
        hdu = fits.open(files[0])
        ssp = hdu[0].data
        h2 = hdu[0].header
        lam = h2["CRVAL1"] + np.arange(h2["NAXIS1"]) * h2["CDELT1"]
        lam_range_temp = lam[[0, -1]]
        ssp_new, log_lam_temp = util.log_rebin(lam_range_temp, ssp, velscale=velscale)[
            :2
        ]

        if norm_range is not None:
            norm_range = np.log(norm_range)
            band = (norm_range[0] <= log_lam_temp) & (log_lam_temp <= norm_range[1])

        templates = np.empty((ssp_new.size, n_ages, n_metal, n_alpha))
        age_grid = np.empty((n_ages, n_metal, n_alpha))
        metal_grid = np.empty((n_ages, n_metal, n_alpha))
        alpha_grid = np.empty((n_ages, n_metal, n_alpha))

        # Convolve the whole Vazdekis library of spectral templates
        # with the quadratic difference between the galaxy and the
        # Vazdekis instrumental resolution. Logarithmically rebin
        # and store each template as a column in the array TEMPLATES.

        # Quadratic sigma difference in pixels Vazdekis --> galaxy
        # The formula below is rigorously valid if the shapes of the
        # instrumental spectral profiles are well approximated by Gaussians.
        if isinstance(FWHM_gal, float):
            FWHM_dif = np.sqrt((FWHM_gal**2 - FWHM_tem**2))
        else:
            FWHM_dif = np.sqrt((FWHM_gal**2 - FWHM_tem**2).clip(0))

        sigma = FWHM_dif / 2.355 / h2["CDELT1"]  # Sigma difference in pixels

        # Here we make sure the spectra are sorted in both [M/H] and Age
        # along the two axes of the rectangular grid of templates.
        for l, alph in enumerate(alphas):
            for j, age in enumerate(ages):
                for k, met in enumerate(metals):
                    p = all.index((age, met, alph))
                    hdu = fits.open(files[p])
                    ssp = hdu[0].data
                    if np.isscalar(FWHM_gal):
                        if sigma > 0.1:  # Skip convolution for nearly zero sigma
                            ssp = ndimage.gaussian_filter1d(ssp, sigma)
                    else:
                        ssp = util.gaussian_filter1d(
                            ssp, sigma
                        )  # convolution with variable sigma
                    ssp_new = util.log_rebin(lam_range_temp, ssp, velscale=velscale)[0]
                    if norm_range is not None:
                        ssp_new /= np.mean(ssp_new[band])
                    templates[:, j, k, l] = ssp_new
                    age_grid[j, k, l] = age
                    metal_grid[j, k, l] = met
                    alpha_grid[j, k, l] = alph

        if age_range is not None:
            # raise NotImplementedError("Not updated the code to use age_range yet!")
            w = (age_range[0] <= age_grid[:, 0, 0]) & (
                age_grid[:, 0, 0] <= age_range[1]
            )
            templates = templates[:, w, :, :]
            age_grid = age_grid[w, :, :]
            metal_grid = metal_grid[w, :, :]
            alpha_grid = alpha_grid[w, ...]
            n_ages, n_metal, n_alpha = age_grid.shape

        if metal_range is not None:
            raise NotImplementedError("Not updated the code to use metal_range yet!")
            w = (metal_range[0] <= metal_grid[0, :]) & (
                metal_grid[0, :] <= metal_range[1]
            )
            templates = templates[:, :, w]
            age_grid = age_grid[:, w]
            metal_grid = metal_grid[:, w]
            n_ages, n_metal = age_grid.shape

        self.templates = templates / np.median(templates)  # Normalize by a scalar
        self.log_lam_temp = log_lam_temp
        self.age_grid = age_grid
        self.alpha_grid = alpha_grid
        self.metal_grid = metal_grid
        self.n_ages = n_ages
        self.n_metal = n_metal
        self.n_alphas = n_alpha

    ###############################################################################
    # MODIFICATION HISTORY:
    #   V1.0.0: Written. Michele Cappellari, Oxford, 1 December 2016
    #   V1.0.1: Use path.realpath() to deal with symbolic links.
    #       Thanks to Sam Vaughan (Oxford) for reporting problems.
    #       MC, Garching, 11 January 2016
    #   V1.0.2: Changed imports for pPXF as a package. MC, Oxford, 16 April 2018
    #   V1.0.3: Removed dependency on cap_readcol. MC, Oxford, 10 May 2018

    def mass_to_light(self, weights, band="V", quiet=False):
        """
        Computes the M/L in a chosen band, given the weights produced
        in output by pPXF. A Salpeter IMF is assumed (slope=1.3).
        The returned M/L includes living stars and stellar remnants,
        but excludes the gas lost during stellar evolution.

        This procedure uses the photometric predictions
        from Vazdekis+12 and Ricciardelli+12
        http://adsabs.harvard.edu/abs/2012MNRAS.424..157V
        http://adsabs.harvard.edu/abs/2012MNRAS.424..172R
        they were downloaded in December 2016 below and are included in pPXF with permission
        http://www.iac.es/proyecto/miles/pages/photometric-predictions/based-on-miuscat-seds.php

        :param weights: pPXF output with dimensions weights[miles.n_ages, miles.n_metal]
        :param band: possible choices are "U", "B", "V", "R", "I", "J", "H", "K" for
            the Vega photometric system and "u", "g", "r", "i" for the SDSS AB system.
        :param quiet: set to True to suppress the printed output.
        :return: mass_to_light in the given band

        """
        assert (
            self.age_grid.shape == self.metal_grid.shape == weights.shape
        ), "Input weight dimensions do not match"

        vega_bands = ["U", "B", "V", "R", "I", "J", "H", "K"]
        sdss_bands = ["u", "g", "r", "i"]
        vega_sun_mag = [5.600, 5.441, 4.820, 4.459, 4.148, 3.711, 3.392, 3.334]
        sdss_sun_mag = [6.55, 5.12, 4.68, 4.57]  # values provided by Elena Ricciardelli

        file_dir = path.dirname(path.realpath(self.pathname))  # path of this procedure

        if band in vega_bands:
            k = vega_bands.index(band)
            sun_mag = vega_sun_mag[k]
            file2 = file_dir + "/Vazdekis2012_ssp_phot_BASTI_UN_v11.0.txt"
        elif band in sdss_bands:
            raise NotImplementedError("No photometry file fpr SDSS filters?")
            k = sdss_bands.index(band)
            sun_mag = sdss_sun_mag[k]
            file2 = file_dir + "/Vazdekis2012_ssp_sdss_miuscat_UN1.30_v9.txt"
        else:
            raise ValueError("Unsupported photometric band")

        file1_alpha_0 = file_dir + "/Vazdekis2012_ssp_mass_BASTI_UN_v11.0_alpha0.0.txt"
        file1_alpha_0p4 = (
            file_dir + "/Vazdekis2012_ssp_mass_BASTI_UN_v11.0_alpha0.4.txt"
        )
        slope1, MH1, Age1, m_no_gas_alpha_0 = np.loadtxt(
            file1_alpha_0, usecols=[1, 2, 3, 5]
        ).T
        slope1, MH1, Age1, m_no_gas_alpha_0p4 = np.loadtxt(
            file1_alpha_0p4, usecols=[1, 2, 3, 5]
        ).T

        N_values = len(slope1)

        slope1 = np.tile(slope1, 2)
        MH1 = np.tile(MH1, 2)
        Age1 = np.tile(Age1, 2)
        alpha1 = np.concatenate(
            (
                np.full(shape=N_values, fill_value=0.0),
                np.full(shape=N_values, fill_value=0.4),
            )
        )
        m_no_gas = np.concatenate((m_no_gas_alpha_0, m_no_gas_alpha_0p4))

        slope2, MH2, Age2, mag = np.loadtxt(file2, usecols=[1, 2, 3, 4 + k]).T
        slope2 = np.tile(slope2, 2)
        MH2 = np.tile(MH2, 2)
        Age2 = np.tile(Age2, 2)
        alpha2 = np.concatenate(
            (
                np.full(shape=N_values, fill_value=0.0),
                np.full(shape=N_values, fill_value=0.4),
            )
        )
        mag = np.concatenate(
            (mag, mag)
        )  # Assume that the alpha abundance has no effect on luminosity!

        # The following loop is a brute force but very safe and general
        # way of matching the photometric quantities to the SSP spectra.
        # It makes no assumption on the sorting and dimensions of the files
        mass_no_gas_grid = np.empty_like(weights)
        lum_grid = np.empty_like(weights)
        for j in range(self.n_ages):
            for k in range(self.n_metal):
                for l in range(self.n_alphas):
                    p1 = (
                        (np.abs(self.age_grid[j, k, l] - Age1) < 0.001)
                        & (np.abs(self.metal_grid[j, k, l] - MH1) < 0.01)
                        & (np.abs(self.alpha_grid[j, k, l] - alpha1) < 0.01)
                        & (np.abs(1.30 - slope1) < 0.01)
                    )
                    mass_no_gas_grid[j, k, l] = m_no_gas[p1]

                    p2 = (
                        (np.abs(self.age_grid[j, k, l] - Age2) < 0.001)
                        & (np.abs(self.metal_grid[j, k, l] - MH2) < 0.01)
                        & (np.abs(self.alpha_grid[j, k, l] - alpha2) < 0.01)
                        & (np.abs(1.30 - slope2) < 0.01)
                    )
                    lum_grid[j, k, l] = 10 ** (-0.4 * (mag[p2] - sun_mag))

        # This is eq.(2) in Cappellari+13
        # http://adsabs.harvard.edu/abs/2013MNRAS.432.1862C
        mlpop = np.sum(weights * mass_no_gas_grid) / np.sum(weights * lum_grid)

        if not quiet:
            print("M/L_" + band + ": %.4g" % mlpop)

        return mlpop

    ###############################################################################

    def plot(self, weights, nodots=False, colorbar=True, **kwargs):
        # assert weights.ndim == 2, "`weights` must be 2-dim"
        # assert self.age_grid.shape == self.metal_grid.shape == weights.shape, \
        #    "Input weight dimensions do not match"

        xgrid = np.log10(self.age_grid) + 9
        ygrid = self.metal_grid
        util.plot_weights_2d(
            xgrid, ygrid, weights, nodots=nodots, colorbar=colorbar, **kwargs
        )

    ##############################################################################

    def mean_age_metal_alpha(self, weights, quiet=False):
        assert weights.ndim == 3, "`weights` must be 2-dim"
        assert (
            self.age_grid.shape
            == self.metal_grid.shape
            == self.alpha_grid.shape
            == weights.shape
        ), "Input weight dimensions do not match"

        log_age_grid = np.log10(self.age_grid) + 9
        metal_grid = self.metal_grid
        alpha_grid = self.alpha_grid

        # These are eq.(1) and (2) in McDermid+15
        # http://adsabs.harvard.edu/abs/2015MNRAS.448.3484M
        mean_log_age = np.sum(weights * log_age_grid) / np.sum(weights)
        mean_metal = np.sum(weights * metal_grid) / np.sum(weights)
        mean_alpha = np.sum(weights * alpha_grid) / np.sum(weights)

        if not quiet:
            print("Weighted <logAge> [yr]: %.3g" % mean_log_age)
            print("Weighted <[M/H]>: %.3g" % mean_metal)
            print("Weighted <[alpha/Fe]>: %.3g" % mean_alpha)

        return mean_log_age, mean_metal, mean_alpha


##############################################################################
