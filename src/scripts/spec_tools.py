import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from specutils import Spectrum1D
import astropy.units as u
from astropy.nddata import StdDevUncertainty
import ppxf.ppxf as ppxf
import ppxf.ppxf_util as util
import scipy.constants as const
import yaml
from tqdm import tqdm
from astropy.io import fits
import glob
import scipy.interpolate as si
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator  # , AutoMinorLocator

from . import spv_miles_util as SAM_lib


class SpectrumCollection:

    """
    A basic class for dealing with a collection of spectra (or a single one).
    """

    def __init__(
        self,
        wavelength,
        flux,
        uncertainty,
        wavelength_units="AA",
        flux_units="erg cm-2 s-1 AA-1",
        uncertainty_is_variance=False,
        meta=None,
        logrebinned=False,
        normalise=False,
        FWHM_gal=2.66,
        settings_filepath=None,
        quiet=False,
    ):
        """
        Initialise a SpectrumColletion class:

        Parameters:
            * wavelength (array, N_lamdas): An array of wavelength values for each flux value
            * flux (array, N_spectra x N_lamdas): A collection of flux spectra
            * uncertainty (array, N_spectra x N_lamdas): A collection of noise spectra. If they're variances, you must also pass 'uncertainty_is_variance=True'
            * wavelength_units (str or astropy.units instance): units of the wavelength values in the spectrum. Can be a string (which will be parsed by astropy.units.Unit) or an astropy units instance
            * flux_units (str or astropy.units instance): units of the flux values in the spectrum. Can be a string (which will be parsed by astropy.units.Unit) or an astropy units instance
            * uncertainty_is_variance (bool): A flag showing the uncertainty values are variances not standard deviations
            * meta (dict): A dictionary to be attached to the Spectrum1D object
            * logrebinned (bool): A flag to state whether the spectra have been logarithmically rebinned (by e.g. ppxf)
            * normalise (bool): Normalise the spectra by the flux median?
            * FWHM_gal (float or array): The FWHM of the flux spectrum. Can either be one value for the whole spectrum or an array of values the same length as the flux array
            * settings_filepath (str): Filename of the text file which contains the location of the templates to use

        """

        # Make sure the flux arrays are 2D
        flux = np.atleast_2d(flux)
        uncertainty = np.atleast_2d(uncertainty)

        # Some basic checks on the flux, wavelentgth and noise arrays
        if flux.shape[1] != len(wavelength):
            raise ValueError(
                "Length of wavelength array must match the second dimension of the flux array (which is N_spectra x N_lam"
            )
        if flux.shape != uncertainty.shape:
            raise ValueError(
                f"Flux and Uncertainty arrays must be the same shape. Currently {flux.shape} != {uncertainty.shape}"
            )

        # Change the units to an astropy units instance if they're a string
        if isinstance(flux_units, str):
            flux_units = u.Unit(flux_units)
        if isinstance(wavelength_units, str):
            wavelength_units = u.Unit(wavelength_units)

        # Add the logrebinned keyword to the meta dictionary (or make it if it doesn't exist)
        if meta is None:
            meta = dict(logrebinned=logrebinned)
        elif isinstance(meta, dict):
            meta["logrebinned"] = logrebinned
        else:
            raise TypeError(
                f"meta type not understood: must be None or dict, not {type(meta)}"
            )

        # Normalise
        if normalise:
            flux_median = np.nanmedian(flux, axis=1)
            if np.any(flux_median <= 0):
                zeros = np.where(flux_median <= 0)[0]
                print(
                    f"Warning: spectra {zeros} have median less than or equal to 0! Fixing now..."
                )
                for z in zeros:
                    flux_median[z] = np.nanmedian(flux[z, flux[z] > 0.0])
            flux /= flux_median[:, None]
            uncertainty /= flux_median[:, None]
            # If we have variances, remember to divide again
            if uncertainty_is_variance:
                uncertainty /= flux_median[:, None]

        # Get the noise spectra as a VarianceUncertainty instance
        if uncertainty_is_variance:
            noise_spectra = StdDevUncertainty(np.sqrt(uncertainty), unit=flux_units)
        else:
            noise_spectra = StdDevUncertainty(uncertainty, unit=flux_units)

        # if settings_filepath is None:
        #     # If the path to the settings file isn't given, check in the directory of this file
        #     self.settings_filepath = f"{os.path.dirname(os.path.realpath(__file__))}/spectral_fit_settings.txt"
        #     if not os.path.exists(self.settings_filepath):
        #         raise FileNotFoundError("No file called spectral_fit_settings.txt")

        self.spectrum = Spectrum1D(
            spectral_axis=wavelength * wavelength_units,
            flux=flux * flux_units,
            uncertainty=noise_spectra,
            meta=meta,
        )

        self.template_pathname = self._set_templates_pathname(
            self.settings_filepath, quiet=quiet
        )

        # Make some attributes which we may use later for spectral fitting
        self.FWHM_gal = FWHM_gal
        self.velscale = None

        self.templates = None
        self.miles = None
        self.reg_dim = None
        self.gas_names = None
        self.line_wave = None
        self.dv = None
        self.ppxf_start = None
        self.n_temps = None
        self.n_forbidden = None
        self.n_balmer = None
        self.component = None
        self.gas_component = None
        self.moments = None
        self.gas_reddening = None

    def _set_templates_pathname(self, filename, templates="MILES", quiet=False):
        with open(filename, "r") as f:
            settings = yaml.safe_load(f)

        return settings["MILES_pathname"]

    # ----------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- PLOTTING -----------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    def plot(self, indices=None, ax=None, **kwargs):
        """
        Plot a single spectrum or a collection of spectra. By default, plot all spectra.
        Parameters:
            * indices (int or list). An integer or list used to index the flux array
            * **kwargs (dict): Passed to ax.plot
        """

        if ax is None:
            fig, ax = plt.subplots()

        wavelength = self.spectrum.spectral_axis

        if indices is not None:
            spectra = self.spectrum.flux[indices, :]
            noise_spectra = self.spectrum.uncertainty[indices, :]
        else:
            spectra = self.spectrum.flux
            noise_spectra = self.spectrum.uncertainty

        self._plot_lines(ax, wavelength, spectra.T, **kwargs)
        self._plot_lines(ax, wavelength, noise_spectra.array.T, c="grey", linewidth=1.5)

        ax.set_xlabel(f"$\\lambda$ ({wavelength.unit})")
        ax.set_ylabel(r"$F_{{\lambda}}$ ({})".format(spectra.unit))
        return ax

    @staticmethod
    def _plot_lines(ax, x, y, **kwargs):
        ln = ax.plot(x, y, **kwargs)
        return ln

    # ----------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- PPXF FITTING -------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    def set_MILES_attributes_for_ppxf_fitting(
        self,
        templates_pathname,
        velscale,
        FWHM_gal,
        lamRange_galaxy,
        tie_balmer=True,
        limit_doublets=True,
        light_weighted=False,
        include_emission_lines=True,
        age_range=None,
        all_lines_free=False,
    ):
        (
            miles,
            templates,
            reg_dim,
            gas_names,
            line_wave,
            dv,
            start,
            n_temps,
            n_forbidden,
            n_balmer,
            component,
            gas_component,
            moments,
            gas_reddening,
        ) = set_up_ppxf_with_MILES(
            templates_pathname,
            velscale=velscale,
            FWHM_gal=FWHM_gal,
            lamRange_galaxy=lamRange_galaxy,
            tie_balmer=tie_balmer,
            limit_doublets=limit_doublets,
            light_weighted=light_weighted,
            galaxy_wavelength_array=self.spectrum.spectral_axis,
            include_emission_lines=include_emission_lines,
            age_range=age_range,
            all_lines_free=all_lines_free,
        )

        self.templates = templates
        self.miles = miles
        self.reg_dim = reg_dim
        self.gas_names = gas_names
        self.line_wave = line_wave
        self.dv = dv
        self.ppxf_start = start
        self.n_temps = n_temps
        self.n_forbidden = n_forbidden
        self.n_balmer = n_balmer
        self.component = component
        self.gas_component = gas_component
        self.moments = moments
        self.gas_reddening = gas_reddening

    def _set_velscale(self, lamRange_galaxy, flux):
        galaxy, logLam_gal, velscale = util.log_rebin(lamRange_galaxy, flux)
        self.velscale = velscale
        return galaxy, logLam_gal, velscale

    def standard_ppxf_stellar_pops_with_MILES_single_spectrum(
        self,
        index,
        z_gal,
        include_emission_lines=True,
        tie_balmer=True,
        limit_doublets=True,
        quiet=True,
        light_weighted=False,
        mask_wavelengths_restframe=[3540.0, 7400.0],
        SAMI_chip_gap_mask=True,
        age_range=None,
        all_lines_free=False,
        **kwargs,
    ):
        """
        A high level function to fully run ppxf on a single spectrum.
        """

        # Mask out things outside the MILES template range
        spectrum = self.spectrum[index, :]

        wavelength_mask, lamRange_galaxy = mask_wavelengths(
            spectrum, z_gal, mask_wavelengths_restframe=mask_wavelengths_restframe
        )

        self.lamRange_galaxy = lamRange_galaxy

        # Logrebin the galaxy and noise templates (if they're not already in log-lamda)
        if not spectrum.meta["logrebinned"]:
            galaxy, logLam_gal, velscale = self._set_velscale(
                lamRange_galaxy, spectrum.flux.value[wavelength_mask]
            )
            noise, logLam_gal, velscale = util.log_rebin(
                lamRange_galaxy, spectrum.uncertainty.array[wavelength_mask]
            )
        else:
            self.velscale = (
                np.log(
                    self.spectrum.spectral_axis[1] / self.spectrum.spectral_axis[0]
                ).value
                * const.c
                / 1000.0
            )
            galaxy = spectrum.flux.value[wavelength_mask]
            noise = spectrum.uncertainty.array[wavelength_mask]
            logLam_gal = np.log(spectrum.spectral_axis.value[wavelength_mask])

        # Get a mask for any bad values in the galaxy or noise
        goodpixel_mask, galaxy, noise = mask_bad_values(galaxy, noise)
        self.ppxf_mask = goodpixel_mask

        # If a SAMI spectrum, mask a bit extra around the chip gap
        if SAMI_chip_gap_mask:
            chip_gap_mask = SAMI_chip_gap_mask(np.exp(logLam_gal), z_gal)
            self.ppxf_mask = self.ppxf_mask & (~chip_gap_mask)

        self.set_MILES_attributes_for_ppxf_fitting(
            self.template_pathname,
            self.velscale,
            self.FWHM_gal,
            self.lamRange_galaxy,
            tie_balmer=tie_balmer,
            limit_doublets=limit_doublets,
            light_weighted=light_weighted,
            include_emission_lines=include_emission_lines,
            age_range=age_range,
            all_lines_free=all_lines_free,
        )

        if include_emission_lines is False:
            lamRangeTemp = [
                np.min(np.exp(self.miles.log_lam_temp)),
                np.max(np.exp(self.miles.log_lam_temp)),
            ]
            good_pixels_list = util.determine_goodpixels(
                logLam_gal, lamRangeTemp, 0.0, width=800
            )

            # Find everything which isn't in the goodpixels list
            things_to_mask = set(np.arange(len(self.ppxf_mask))) - set(good_pixels_list)
            self.ppxf_mask[list(things_to_mask)] = False

        pp = self.run_ppxf(
            self.templates,
            galaxy,
            noise,
            velscale=self.velscale,
            start=self.ppxf_start,
            dv=self.dv,
            moments=self.moments,
            lam=np.exp(logLam_gal),
            reg_dim=self.reg_dim,
            component=self.component,
            gas_component=self.gas_component,
            gas_names=self.gas_names,
            gas_reddening=self.gas_reddening,
            mask=self.ppxf_mask,
            **kwargs,
        )

        return pp

    def standard_ppxf_stellar_pops_with_MILES_all_spectra(
        self,
        z_gal,
        include_emission_lines=True,
        tie_balmer=True,
        limit_doublets=True,
        quiet=True,
        light_weighted=False,
        SAMI_chip_gap_mask=True,
        mask_wavelengths_restframe=[3540.0, 7400.0],
        clean=False,
        age_range=None,
        all_lines_free=False,
        miles_or_bpass="miles",
        **kwargs,
    ):
        """
        A high level function to fully run ppxf on a set of spectra (without having to load all the templates again)
        """

        # Mask out things outside the MILES template range
        wavelength_mask, lamRange_galaxy = mask_wavelengths(
            self.spectrum[0, :],
            z_gal,
            mask_wavelengths_restframe=mask_wavelengths_restframe,
        )

        self.lamRange_galaxy = lamRange_galaxy

        # Logrebin the galaxy and noise templates (if they're not already in log-lamda)
        if not self.spectrum.meta["logrebinned"]:
            galaxy, logLam_gal, velscale = self._set_velscale(
                lamRange_galaxy, self.spectrum.flux.value[0, wavelength_mask]
            )
        else:
            self.velscale = (
                np.log(
                    self.spectrum.spectral_axis[1] / self.spectrum.spectral_axis[0]
                ).value
                * const.c
                / 1000.0
            )
            logLam_gal = np.log(self.spectrum.spectral_axis.value[wavelength_mask])

        if miles_or_bpass == "miles":
            print("Using MILES templates")
            self.set_MILES_attributes_for_ppxf_fitting(
                self.template_pathname,
                self.velscale,
                self.FWHM_gal,
                self.lamRange_galaxy,
                tie_balmer=tie_balmer,
                limit_doublets=limit_doublets,
                light_weighted=light_weighted,
                include_emission_lines=include_emission_lines,
                age_range=age_range,
                all_lines_free=all_lines_free,
            )
        else:
            raise NotImplementedError("Only works with the MILES templates for now")

        N_spectra = self.spectrum.shape[0]
        pp_objects = []
        for i in range(N_spectra):
            # Get a mask for any bad values in the galaxy or noise
            f = self.spectrum.flux.value[i, wavelength_mask]
            n = self.spectrum.uncertainty.array[i, wavelength_mask]

            # Logrebin the galaxy and noise templates (if they're not already in log-lamda)
            if not self.spectrum.meta["logrebinned"]:
                galaxy, logLam_gal, velscale = self._set_velscale(lamRange_galaxy, f)
                noise, logLam_gal, velscale = util.log_rebin(lamRange_galaxy, n)

            else:
                self.velscale = (
                    np.log(
                        self.spectrum.spectral_axis[1] / self.spectrum.spectral_axis[0]
                    ).value
                    * const.c
                    / 1000.0
                )
                galaxy = f
                noise = n
                logLam_gal = np.log(self.spectrum.spectral_axis.value[wavelength_mask])

            goodpixel_mask, galaxy, noise = mask_bad_values(galaxy, noise)
            if SAMI_chip_gap_mask:
                chip_gap_mask = SAMI_chip_gap_mask(np.exp(logLam_gal), z_gal)
                goodpixel_mask = goodpixel_mask & (~chip_gap_mask)

            if clean:
                # If we want to use CLEAN, fit once to get an idea of the chi2 and then rescale the noise by this
                pp = self.run_ppxf(
                    self.templates,
                    galaxy,
                    noise,
                    velscale=self.velscale,
                    start=self.ppxf_start,
                    dv=self.dv,
                    moments=self.moments,
                    lam=np.exp(logLam_gal),
                    reg_dim=self.reg_dim,
                    component=self.component,
                    gas_component=self.gas_component,
                    gas_names=self.gas_names,
                    gas_reddening=self.gas_reddening,
                    mask=goodpixel_mask,
                    clean=False,
                    **kwargs,
                )
                noise *= np.sqrt(pp.chi2)

            pp = self.run_ppxf(
                self.templates,
                galaxy,
                noise,
                velscale=self.velscale,
                start=self.ppxf_start,
                dv=self.dv,
                moments=self.moments,
                lam=np.exp(logLam_gal),
                reg_dim=self.reg_dim,
                component=self.component,
                gas_component=self.gas_component,
                gas_names=self.gas_names,
                gas_reddening=self.gas_reddening,
                mask=goodpixel_mask,
                clean=clean,
                **kwargs,
            )

            pp_objects.append(pp)

        if N_spectra == 1:
            return pp
        return pp_objects

    def bootstrap_spectrum_with_MILES(
        self,
        index,
        z_gal,
        N_bootstraps,
        chunk_length=100,
        include_emission_lines=True,
        tie_balmer=True,
        limit_doublets=True,
        quiet=True,
        light_weighted=False,
        SAMI_chip_gap_mask=True,
        mask_wavelengths_restframe=[3540.0, 7400.0],
        clean=False,
        age_range=None,
        **kwargs,
    ):
        """
        A high level function to fully run ppxf on a set of spectra (without having to load all the templates again)
        """

        # Make a wavelength mask to chop out things outside the MILES template range
        wavelength_mask, lamRange_galaxy = mask_wavelengths(
            self.spectrum[index, :],
            z_gal,
            mask_wavelengths_restframe=mask_wavelengths_restframe,
        )

        self.lamRange_galaxy = lamRange_galaxy

        # Logrebin the galaxy and noise templates (if they're not already in log-lamda) to get the velscale
        if not self.spectrum.meta["logrebinned"]:
            galaxy, logLam_gal, velscale = self._set_velscale(
                lamRange_galaxy, self.spectrum.flux.value[0, wavelength_mask]
            )
        else:
            self.velscale = (
                np.log(
                    self.spectrum.spectral_axis[1] / self.spectrum.spectral_axis[0]
                ).value
                * const.c
                / 1000.0
            )
            logLam_gal = np.log(self.spectrum.spectral_axis.value[wavelength_mask])

        self.set_MILES_attributes_for_ppxf_fitting(
            self.template_pathname,
            self.velscale,
            self.FWHM_gal,
            self.lamRange_galaxy,
            tie_balmer=tie_balmer,
            limit_doublets=limit_doublets,
            light_weighted=light_weighted,
            include_emission_lines=include_emission_lines,
            age_range=age_range,
        )

        # Get a mask for any bad values in the galaxy or noise
        f = self.spectrum.flux.value[index, wavelength_mask]
        n = self.spectrum.uncertainty.array[index, wavelength_mask]

        # Logrebin the galaxy and noise templates (if they're not already in log-lamda)
        if not self.spectrum.meta["logrebinned"]:
            galaxy, logLam_gal, velscale = self._set_velscale(lamRange_galaxy, f)
            noise, logLam_gal, velscale = util.log_rebin(lamRange_galaxy, n)

        else:
            self.velscale = (
                np.log(
                    self.spectrum.spectral_axis[1] / self.spectrum.spectral_axis[0]
                ).value
                * const.c
                / 1000.0
            )
            galaxy = f
            noise = n
            logLam_gal = np.log(self.spectrum.spectral_axis.value[wavelength_mask])

        goodpixel_mask, galaxy, noise = mask_bad_values(galaxy, noise)
        if SAMI_chip_gap_mask:
            chip_gap_mask = SAMI_chip_gap_mask(np.exp(logLam_gal), z_gal)
            goodpixel_mask = goodpixel_mask & (~chip_gap_mask)

        pp = self.run_ppxf(
            self.templates,
            galaxy,
            noise,
            velscale=self.velscale,
            start=self.ppxf_start,
            dv=self.dv,
            moments=self.moments,
            lam=np.exp(logLam_gal),
            reg_dim=self.reg_dim,
            component=self.component,
            gas_component=self.gas_component,
            gas_names=self.gas_names,
            gas_reddening=self.gas_reddening,
            mask=goodpixel_mask,
            clean=clean,
            **kwargs,
        )

        bestfit = pp.bestfit
        residuals = pp.galaxy - bestfit

        pp_objects = []
        for i in tqdm(range(N_bootstraps)):
            shuffled_residuals = self.shuffle_residuals_in_chunks(
                residuals, chunk_length=chunk_length, wavelength_mask=goodpixel_mask
            )
            galaxy = bestfit + shuffled_residuals

            if clean:
                # If we want to use CLEAN, fit once to get an idea of the chi2 and then rescale the noise by this
                pp = self.run_ppxf(
                    self.templates,
                    galaxy,
                    noise,
                    velscale=self.velscale,
                    start=self.ppxf_start,
                    dv=self.dv,
                    moments=self.moments,
                    lam=np.exp(logLam_gal),
                    reg_dim=self.reg_dim,
                    component=self.component,
                    gas_component=self.gas_component,
                    gas_names=self.gas_names,
                    gas_reddening=self.gas_reddening,
                    mask=goodpixel_mask,
                    clean=False,
                    **kwargs,
                )
                noise *= np.sqrt(pp.chi2)

            pp = self.run_ppxf(
                self.templates,
                galaxy,
                noise,
                velscale=self.velscale,
                start=self.ppxf_start,
                dv=self.dv,
                moments=self.moments,
                lam=np.exp(logLam_gal),
                reg_dim=self.reg_dim,
                component=self.component,
                gas_component=self.gas_component,
                gas_names=self.gas_names,
                gas_reddening=self.gas_reddening,
                mask=goodpixel_mask,
                clean=clean,
                **kwargs,
            )

            pp_objects.append(pp)

        return pp_objects

    def run_ppxf(self, templates, galaxy, noise, velscale, start, dv, **kwargs):
        pp = ppxf.ppxf(templates, galaxy, noise, velscale, start, vsyst=dv, **kwargs)

        return pp

    @staticmethod
    def shuffle_residuals_in_chunks(
        residuals, chunk_length, wavelength_mask, order="wavelength"
    ):
        """
        Take a set of residuals, break into chunks and then shuffle those chunks
        This is made much more complicated when we have a mask in the middle of the residuals.
        To sort this, we also provide a wavelength mask. We then chunk up that mask too and
        see if many of the elements in a chunk are masked. If so, we skip that chunk
        """

        masked_residuals = ma.array(residuals, mask=~wavelength_mask)
        N_res = len(masked_residuals)
        N_chunks = N_res // chunk_length
        remainder = N_res % chunk_length

        # Reshape the residuals into their chunks
        # This if is needed since array[:-0] results in a 0-length array
        if remainder != 0:
            reshaped_residuals = masked_residuals[:-remainder].reshape(
                N_chunks, chunk_length
            )
            reshaped_mask = wavelength_mask[:-remainder].reshape(N_chunks, chunk_length)
        else:
            reshaped_residuals = masked_residuals.reshape(N_chunks, chunk_length)
            reshaped_mask = wavelength_mask.reshape(N_chunks, chunk_length)

        # These are all chunks which aren't completely masked
        good_chunks = np.where(reshaped_mask.mean(axis=1) > 0.0)[0]

        final_residuals = np.zeros_like(reshaped_residuals)

        if order == "random":
            for i in range(N_chunks):
                res_order = np.random.choice(
                    np.arange(chunk_length), chunk_length, replace=False
                )
                final_residuals[i, :] = reshaped_residuals[i, res_order]

        elif order == "wavelength":
            for good_chunk, random_chunk in zip(
                good_chunks,
                np.random.choice(good_chunks, size=len(good_chunks), replace=False),
            ):
                final_residuals[good_chunk, :] = reshaped_residuals[random_chunk, :]
        else:
            raise NameError("Order must be random or wavelength")

        shuffled_residuals = final_residuals.flatten().filled(0.0)

        # shuffled_residuals = final_residuals.flatten()

        # And now shuffle the remaining residuals and add to the end (if our original array isn't a multiple of chunk length)
        if remainder > 0:
            shuffled_residuals = np.append(
                shuffled_residuals,
                np.random.choice(
                    masked_residuals[-remainder:].filled(
                        masked_residuals[-remainder:].mean()
                    ),
                    remainder,
                    replace=False,
                ),
            )

        return shuffled_residuals

    # def standard_ppxf_stellar_pops_with_MILES_all_spectra(self, z_gal, tie_balmer=True, limit_doublets=True, quiet=True, light_weighted=False, SAMI_chip_gap_mask=True, **kwargs):
    #     """
    #     A high level function to fully run ppxf on a set of spectra (without having to load all the templates again). kwargs are passed to ppxf.ppxf
    #     """

    #     # Mask out things outside the MILES template range
    #     wavelength_mask, lamRange_galaxy = mask_wavelengths(self.spectrum[0, :], z_gal, mask_wavelengths_restframe=[3540.0, 7400.0])

    #     self.lamRange_galaxy = lamRange_galaxy

    #     # Logrebin the galaxy and noise templates (if they're not already in log-lamda)
    #     if not self.spectrum.meta['logrebinned']:
    #         galaxy, logLam_gal, velscale = self._set_velscale(lamRange_galaxy, self.spectrum.flux.value[0, wavelength_mask])
    #     else:
    #         self.velscale = np.log(self.spectrum.spectral_axis[1]/self.spectrum.spectral_axis[0]).value * const.c/1000.0
    #         logLam_gal = np.log(self.spectrum.spectral_axis.value[wavelength_mask])

    #     self.set_MILES_attributes_for_ppxf_fitting(self.template_pathname, self.velscale, self.FWHM_gal, self.lamRange_galaxy, tie_balmer=tie_balmer, limit_doublets=limit_doublets, light_weighted=light_weighted)

    #     N_spectra = self.spectrum.shape[0]
    #     pp_objects = []
    #     for i in range(N_spectra):

    #         # Get a mask for any bad values in the galaxy or noise
    #         goodpixel_mask, f, n = mask_bad_values(self.spectrum.flux.value[i, wavelength_mask], self.spectrum.uncertainty.array[i, wavelength_mask])

    #         if SAMI_chip_gap_mask:
    #             chip_gap_mask = SAMI_chip_gap_mask(np.exp(logLam_gal), z_gal)
    #             goodpixel_mask = goodpixel_mask & (~chip_gap_mask)

    #         # Logrebin the galaxy and noise templates (if they're not already in log-lamda)
    #         if not self.spectrum.meta['logrebinned']:
    #             galaxy, logLam_gal, velscale = self._set_velscale(lamRange_galaxy, f)
    #             noise, logLam_gal, velscale = util.log_rebin(lamRange_galaxy, n)
    #         else:
    #             self.velscale = np.log(self.spectrum.spectral_axis[1] / self.spectrum.spectral_axis[0]).value * const.c / 1000.0
    #             galaxy = f
    #             noise = n
    #             logLam_gal = np.log(self.spectrum.spectral_axis.value[wavelength_mask])

    #         pp = self.run_ppxf(self.templates, galaxy, noise, velscale=self.velscale, start=self.ppxf_start, dv=self.dv, moments=self.moments, lam=np.exp(logLam_gal), reg_dim=self.reg_dim, component=self.component, gas_component=self.gas_component, gas_names=self.gas_names, gas_reddening=self.gas_reddening, mask=goodpixel_mask, **kwargs)

    #         pp_objects.append(pp)

    #     if N_spectra == 1:
    #         return pp
    #     return pp_objects


def set_up_ppxf_with_MILES(
    MILES_pathname,
    velscale,
    FWHM_gal,
    lamRange_galaxy,
    galaxy_wavelength_array,
    tie_balmer=True,
    limit_doublets=True,
    light_weighted=False,
    quiet=True,
    include_emission_lines=True,
    age_range=None,
    all_lines_free=False,
):
    (
        miles,
        stars_templates,
        gas_templates,
        reg_dim,
        gas_names,
        line_wave,
    ) = _load_MILES_templates(
        MILES_pathname=MILES_pathname,
        velscale=velscale,
        FWHM_gal=FWHM_gal,
        lamRange_galaxy=lamRange_galaxy,
        tie_balmer=tie_balmer,
        limit_doublets=limit_doublets,
        quiet=quiet,
        light_weighted=light_weighted,
        galaxy_wavelength_array=galaxy_wavelength_array,
        age_range=age_range,
    )

    if include_emission_lines:
        templates = np.column_stack([stars_templates, gas_templates])
    else:
        templates = stars_templates

    (
        dv,
        start,
        n_temps,
        n_forbidden,
        n_balmer,
        component,
        gas_component,
        moments,
        gas_reddening,
    ) = _make_standard_ppxf_args(
        lamRange_galaxy=lamRange_galaxy,
        miles=miles,
        tie_balmer=tie_balmer,
        stars_templates=stars_templates,
        gas_names=gas_names,
        include_emission_lines=include_emission_lines,
        all_lines_free=all_lines_free,
    )

    return (
        miles,
        templates,
        reg_dim,
        gas_names,
        line_wave,
        dv,
        start,
        n_temps,
        n_forbidden,
        n_balmer,
        component,
        gas_component,
        moments,
        gas_reddening,
    )


def _load_MILES_templates(
    MILES_pathname,
    velscale,
    FWHM_gal,
    lamRange_galaxy,
    galaxy_wavelength_array,
    tie_balmer=True,
    limit_doublets=True,
    quiet=False,
    light_weighted=False,
    include_emission_lines=True,
    age_range=None,
):
    """
    Load the MILES stellar template library, using my modified version of Michele's miles library class. We also load the gas templates corresponding to the gaalxy wavelength range
    Parameters:
        * MILES_pathname (str): Location of the MILES templates. Must be a glob matching the fits files
        * velscale (float): Velocity difference between adjacent pixels of the (log-rebinned) flux array
        * FWHM_gal (float): FWHM of the flux array. The MILES templates are convolved up to this FWHM. TODO: make this able to be a function/array
        * lamRange_galaxy (list): The start and end wavelengths of the flux spectrum
        tie_balmer (bool): Force the balmer decrement for Balmer emission lines? Must be true if fitting for gas redenning
        * limit_doublets (bool): Limit emission line doublets to their theoretical values?
        * quiet (bool): Print info or not
        light_weighted (bool): Normalise the templates between [5070, 5950] Angstroms (for light-weighted results) or not (for mass weighted results)
    Returns:
        (miles, stars_templates, gas_templates, reg_dim, gas_names, line_wave)
    """

    if light_weighted:
        norm_range = [5070, 5950]
    else:
        norm_range = None

    # If the FWHM is an array, we need to make sure that we have an FWHM value at every wavelength pixel of the _template_ array. We do this by interpolation. By definition, this will involve some extrapolation- but this dosesn't matter since we'll make out regions where the templates are outside the galaxy spectrum anyway
    if hasattr(FWHM_gal, "__len__"):
        # Load the templates once with a single scalar value to get the template wavelength array
        t_hdu = fits.open(glob.glob(MILES_pathname)[0])
        t_header = t_hdu[0].header
        lam_temp = t_header["CRVAL1"] + t_header["CDELT1"] * np.arange(
            t_header["NAXIS1"]
        )
        FWHM_interpolator = si.interp1d(
            x=galaxy_wavelength_array,
            y=FWHM_gal,
            bounds_error=False,
            fill_value=(FWHM_gal[0], FWHM_gal[-1]),
        )
        FWHM_gal = FWHM_interpolator(lam_temp)

    miles = SAM_lib.miles(
        MILES_pathname, velscale, FWHM_gal, norm_range=norm_range, age_range=age_range
    )

    # The stellar templates are reshaped below into a 2-dim array with each
    # spectrum as a column, however we save the original array dimensions,
    # which are needed to specify the regularization dimensions
    #
    reg_dim = miles.templates.shape[1:]
    stars_templates = miles.templates.reshape(miles.templates.shape[0], -1)

    # Unfortunately, util.emission_lines expects a scalar or a *function* which returns the FWHM at every wavelength. Make a quick lambda function using the interpolator from before to achieve this.
    if hasattr(FWHM_gal, "__len__"):
        FWHM_gal = lambda x: FWHM_interpolator(x)

    # Construct a set of Gaussian emission line templates.
    # Estimate the wavelength fitted range in the rest frame.
    gas_templates, gas_names, line_wave = util.emission_lines(
        miles.log_lam_temp,
        lamRange_galaxy,
        FWHM_gal,
        tie_balmer=tie_balmer,
        limit_doublets=limit_doublets,
    )

    return miles, stars_templates, gas_templates, reg_dim, gas_names, line_wave


def _make_standard_ppxf_args(
    lamRange_galaxy,
    miles,
    tie_balmer,
    stars_templates,
    gas_names,
    include_emission_lines=True,
    all_lines_free=False,
):
    """
    Set up standard values for ppxf fitting with SAMI. Returns dv, start, n_temps, n_forbidden, n_balmer, component, gas_component, moments, gas_reddening
    """

    # Speed of light in km/s
    c = const.c / 1000.0
    dv = c * (
        miles.log_lam_temp[0] - np.log(lamRange_galaxy[0])
    )  # eq.(8) of Cappellari (2017)
    # vel = c*np.log(1 + z)   # eq.(8) of Cappellari (2017)
    start = [0.0, 100]

    n_temps = stars_templates.shape[1]
    if include_emission_lines:
        n_balmer = np.sum(
            ["Balmer" in a for a in gas_names]
        )  # forbidden lines contain "[*]"
        n_forbidden = len(gas_names) - n_balmer
    else:
        n_forbidden = 0
        n_balmer = 0

    # Assign component=0 to the stellar templates, component=1 to the Balmer
    # gas emission lines templates and component=2 to the forbidden lines.
    if not include_emission_lines:
        component = [0] * n_temps
        gas_component = None
    else:
        if tie_balmer:
            component = [0] * n_temps + [1] * n_balmer + [2] * n_forbidden
        elif all_lines_free is True:
            component = (
                [0] * n_temps
                + list(range(1, n_balmer + 1))
                + list(range(n_balmer + 1, n_balmer + n_forbidden + 1))
            )
        else:
            component = (
                [0] * n_temps
                + list(range(1, n_balmer + 1))
                + [n_balmer + 1] * n_forbidden
            )
        gas_component = np.array(component) > 0  # gas_component=True for gas templates

    # import ipdb; ipdb.set_trace()
    # Fit (V, sig) moments=2 for the stars
    # and (V, sig) moments=2 for the two gas kinematic components
    if include_emission_lines:
        if tie_balmer:
            moments = [2, 2, 2]
        else:
            moments = [2] * len(np.unique(component))
    else:
        moments = [2] * len(np.unique(component))

    if include_emission_lines:
        # Adopt the same starting value for the stars and the two gas components
        start = [start] * len(np.unique(component))

    if include_emission_lines:
        # If the Balmer lines are tied one should allow for gas reddeining.
        # The gas_reddening can be different from the stellar one, if both are fitted.
        gas_reddening = 0 if tie_balmer else None
    else:
        gas_reddening = None

    return (
        dv,
        start,
        n_temps,
        n_forbidden,
        n_balmer,
        component,
        gas_component,
        moments,
        gas_reddening,
    )


def mask_wavelengths(spectra, z, mask_wavelengths_restframe):
    """
    Deredshift a spectrum and mask out all values which don't fall between mask_wavelengths[0] and mask_wavelengths[1].
    NOTE THAT MASK_WAVELENGTH SHOULD BE REST_FRAME VALUES!
    """

    # deredshift the spectrum and mask things which are too blue for the templates
    observed_lamdas = spectra.spectral_axis.value
    wavelength_mask = (observed_lamdas > mask_wavelengths_restframe[0] * (1 + z)) & (
        observed_lamdas < mask_wavelengths_restframe[1] * (1 + z)
    )

    # Mask the data
    lamdas = observed_lamdas[wavelength_mask]

    # Wavelength range of the galaxy
    lamRange_galaxy = np.array([lamdas[0], lamdas[-1]]) / (1 + z)
    # logger.info("LamRange galaxy is {}".format(self.lamRange_galaxy))

    return wavelength_mask, lamRange_galaxy


def mask_bad_values(flux, noise):
    good_values = (
        (np.isfinite(flux)) & (np.isfinite(noise)) & (noise > 0) & (flux > 0.0)
    )

    flux[~good_values] = 0.0
    noise[~good_values] = 1.0

    return good_values, flux, noise


def SAMI_chip_gap_mask(lamdas, z_gal):
    """
    Get a mask of things around the SAMI chip gap. True means in the chip gap
    """

    mask = (lamdas > 5700 / (1 + z_gal)) & (lamdas < 6300 / (1 + z_gal))

    return mask


def analyse_ppxf_output(all_weights, miles, ML_band="r", quiet=True):
    """
    Return order is:

    np.column_stack((all_ages, all_metallicities, all_MLs, all_fractions_last_Gyr, all_fractions_last_10Gyr, all_fractions_less_solar))

    """

    # miles = SAM_lib.miles(spectral_fitting.MILES_pathname, 50, 10)  # dummy numbers for velscale and FWHM

    assert all_weights.ndim == 4, "Must have 4D weights array!"

    ages = np.log10(miles.age_grid) + 9
    metallicities = miles.metal_grid
    alphas = miles.alpha_grid

    n_bins = all_weights.shape[0]

    all_ages = np.zeros(n_bins)
    all_metallicities = np.zeros(n_bins)
    all_alphas = np.zeros(n_bins)
    all_fractions_last_Gyr = np.zeros(n_bins)
    all_fractions_last_10Gyr = np.zeros(n_bins)
    all_fractions_less_solar = np.zeros(n_bins)

    # Loop through the bins
    for n in range(n_bins):
        weights = all_weights[n, :]
        age_metal_weights = np.sum(weights, axis=-1)
        age, M, alpha = miles.mean_age_metal_alpha(weights, quiet=quiet)

        # ML = miles.mass_to_light(weights, band=ML_band, quiet=quiet)

        # Remember we're in log years!
        fraction_last_Gyr = np.sum(weights[ages <= 9]) / np.sum(weights)
        fraction_last_10Gyr = np.sum(weights[ages <= 10]) / np.sum(weights)

        fraction_less_solar = np.sum(weights[metallicities <= 0.0]) / np.sum(weights)

        all_ages[n] = age
        all_metallicities[n] = M
        all_alphas[n] = alpha
        all_fractions_last_Gyr[n] = fraction_last_Gyr
        all_fractions_last_10Gyr[n] = fraction_last_10Gyr
        all_fractions_less_solar[n] = fraction_less_solar

    return np.column_stack(
        (
            all_ages,
            all_metallicities,
            all_alphas,
            all_fractions_last_Gyr,
            all_fractions_last_10Gyr,
            all_fractions_less_solar,
        )
    )


def reshape_weights(weights, reg_dim, gas_component=None):
    if gas_component is not None:
        weights = weights[~gas_component]  # Exclude weights of the gas templates
    normed_weights = weights.reshape(reg_dim) / weights.sum()  # Normalized

    return normed_weights


def results_to_2d(bin_mask, quantity):
    """
    Take a 1D list of results and turn them into a 2D map using a bin mask
    """

    assert (
        len(quantity) == len(np.unique(bin_mask)) - 1
    ), "Need to have the same number of bins as measurements"
    results_map = bin_mask.copy().astype(float)

    quantity = np.insert(quantity, 0, np.nan)
    for i, bin_num in enumerate(np.unique(bin_mask)):
        results_map[results_map == bin_num] = quantity[i]

    return results_map


def save_galaxy_results(
    filename,
    bin_mask,
    ages,
    metallicities,
    alphas,
    fraction_last_Gyr,
    reddening,
    **kwargs,
):
    # bin_map = self.results_to_2d(self.bin_mask_of_ppxf_spectra, self.bin_mask_of_ppxf_spectra)
    age_map = results_to_2d(bin_mask, ages)
    metal_map = results_to_2d(bin_mask, metallicities)
    alpha_map = results_to_2d(bin_mask, alphas)
    fraction_last_Gyr_map = results_to_2d(bin_mask, fraction_last_Gyr)
    gas_reddening_map = results_to_2d(bin_mask, reddening)

    hdu1 = fits.PrimaryHDU()
    # hdu_bins = fits.ImageHDU(bin_map)
    # hdu_bins.header['QUANTITY'] = 'VoronoiBins'

    hdu_age = fits.ImageHDU(age_map, name="LogAge")
    hdu_age.header["QUANTITY"] = "LogAge"

    hdu_metals = fits.ImageHDU(metal_map, name="Metallicity")
    hdu_metals.header["QUANTITY"] = "METALLICITY"

    hdu_alpha = fits.ImageHDU(alpha_map, name="Alpha/Fe")
    hdu_alpha.header["QUANTITY"] = "ALPHA"

    hdu_fraction = fits.ImageHDU(fraction_last_Gyr_map, name="LastGyrFraction")
    hdu_fraction.header["QUANTITY"] = "Fraction_stars_formed_last_Gyr"

    hdu_reddening = fits.ImageHDU(gas_reddening_map, name="Gas_Reddening")
    hdu_fraction.header["QUANTITY"] = "GAS_REDDENING"

    hdu_list = fits.HDUList(
        [hdu1, hdu_age, hdu_metals, hdu_alpha, hdu_fraction, hdu_reddening]
    )
    hdu_list.writeto(filename, **kwargs)


def plot_galaxy_results(
    unbinned_data,
    bin_mask,
    ages,
    metallicities,
    alphas,
    fraction_last_Gyr,
    title="",
    **kwargs,
):
    """
    Make a nice plot of our results for a single galaxy.
    Kwargs go to imshow. By default, the things to show are the galaxy ages, metallicties and fractions of stars formed in the last Gyr.
    """

    # Get the header for WCS info
    # wcs = WCS(self.header).celestial

    fig, axs = plt.subplots(
        nrows=1, ncols=5, figsize=(20, 5)
    )  # ,subplot_kw=dict(projection=wcs), figsize=(15, 5), sharex=True, sharey=True)

    axs[0].imshow(np.nanmedian(unbinned_data, axis=0), cmap="RdGy_r", **kwargs)
    axs[0].set_title("Median cube", fontsize=20, pad=15)

    # age_map=results_to_2d(bin_mask_of_ppxf_spectra, ages)
    # im_age=axs[2].imshow(age_map)
    # divider = make_axes_locatable(axs[2])
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # cbar_age=fig.colorbar(im_age, cax=cax, orientation='vertical')
    # cbar_age.set_label('Gyr')

    axs[1], cbar_age = _make_subplot(
        fig, axs[1], bin_mask, ages, cbar_label="log (years)"
    )
    axs[1].set_title("Log Age", fontsize=20, pad=15)
    axs[2], cbar_metals = _make_subplot(
        fig, axs[2], bin_mask, metallicities, cbar_label="[M/H]"
    )
    axs[2].set_title("[M/H]", fontsize=20, pad=15)
    axs[3], cbar_alphas = _make_subplot(
        fig, axs[3], bin_mask, alphas, cbar_label=r"[$\alpha$/Fe]"
    )
    axs[3].set_title(r"[$\alpha$/Fe]", fontsize=20, pad=15)

    log_fraction = np.log10(fraction_last_Gyr)
    log_fraction[np.isnan(log_fraction)] = -5
    log_fraction[log_fraction < -5] = -5
    axs[4], cbar_frac = _make_subplot(
        fig,
        axs[4],
        bin_mask,
        np.log10(fraction_last_Gyr),
        cbar_label="log(Fraction)",
        vmin=-5.0,
        vmax=0.0,
        cmap="coolwarm_r",
    )
    axs[4].set_title(r"log(Fraction of stars\\formed $<$1Gyr ago)", fontsize=20, pad=15)

    for ax in axs:
        ax.xaxis.set_minor_locator(MultipleLocator(2))
        ax.yaxis.set_minor_locator(MultipleLocator(2))

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_ticks_position("both")
        ax.tick_params(which="both", width=2)
        # ax.tick_params(which='major', length=7)
        # ax.tick_params(which='minor', length=4)

    fig.suptitle(title)
    fig.tight_layout()

    return fig


def _make_subplot(
    fig, axis, bin_mask, quantity, cbar_label="", cmap="RdYlBu_r", fontsize=15, **kwargs
):
    """
    Take a set of 1d results, make a map and plot it with imshow

    """

    quantity_map = results_to_2d(bin_mask, quantity)
    image = axis.imshow(quantity_map, cmap=cmap, **kwargs)
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    cbar = fig.colorbar(image, cax=cax, orientation="vertical")
    cbar.set_label(cbar_label, labelpad=0, fontsize=fontsize)

    return axis, cbar


def shuffle_residuals_in_chunks(residuals, chunk_length):
    """
    Take a set of residuals and shuffle them in chunks of length chunk_length (pixels)
    """

    N_res = len(residuals)
    N_chunks = N_res // chunk_length
    remainder = N_res % chunk_length

    # Reshape the residuals into their chunks
    reshaped_residuals = residuals[:-remainder].reshape(N_chunks, chunk_length)

    final_residuals = np.empty_like(reshaped_residuals)

    for i in range(N_chunks):
        random_order = np.random.choice(
            np.arange(chunk_length), chunk_length, replace=False
        )
        final_residuals[i, :] = reshaped_residuals[i, random_order]

    shuffled_residuals = final_residuals.flatten()

    # And now shuffle the remaining residuals and add to the end (if our original array isn't a multiple of chunk length)
    if remainder > 0:
        shuffled_residuals = np.append(
            shuffled_residuals,
            np.random.choice(residuals[-remainder:], remainder, replace=False),
        )

    return shuffled_residuals
