import scipy.ndimage as nd
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np

import astropy.units as u
import astropy.constants as const
from astropy import log

from specutils import Spectrum1D


class RotationalBroadening(object):
    vrot = 1. * u.km / u.s
    resolution = (20 * u.km / u.s / const.c).to(1)
    limb_darkening = 0.6
    parameters = ['vrot']

    def rotational_profile(self):
        vrot_by_c = (np.maximum(0.1 * u.m / u.s, np.abs(self.vrot)) /
                     const.c).to(1)
        half_width = np.round(vrot_by_c / self.resolution).astype(int)
        profile_velocity = np.linspace(-half_width, half_width,
                                       2 * half_width + 1) * self.resolution
        profile = np.maximum(0.,
                             1. - (profile_velocity / vrot_by_c) ** 2)
        profile = ((2 * (1-self.limb_darkening) * np.sqrt(profile) +
                    0.5 * np.pi * self.limb_darkening * profile) /
                   (np.pi * vrot_by_c * (1.-self.limb_darkening/3.)))
        return profile/profile.sum()

    def __call__(self, spectrum):
        wavelength, flux = spectrum.wavelength.value, spectrum.flux
        log_grid_log_wavelength = np.arange(np.log(wavelength.min()),
                                            np.log(wavelength.max()),
                                            self.resolution.to(1).value)
        log_grid_wavelength = np.exp(log_grid_log_wavelength)
        log_grid_flux = np.interp(log_grid_wavelength, wavelength, flux)
        profile = self.rotational_profile()
        log_grid_convolved = nd.convolve1d(log_grid_flux, profile)
        convolved_flux = np.interp(wavelength, log_grid_wavelength,
                                   log_grid_convolved)
        return Spectrum1D.from_array(spectrum.wavelength,
                                     convolved_flux,
                                     dispersion_unit=spectrum.wavelength.unit,
                                     unit=spectrum.unit)


class DopplerShift(object):

    vrad = 0. * u.km / u.s
    parameters = ['vrad']

    def __call__(self, spectrum):
        doppler_factor = 1. + self.vrad / const.c
        return Spectrum1D.from_array(spectrum.wavelength * doppler_factor,
                                     spectrum.flux,
                                     dispersion_unit=spectrum.wavelength.unit)


#Guassian Convolution
class Convolve(object):
    
    """
    This class can be called to do a gaussian convolution on a given spectrum. You must initialize it with the desired instrumental resolution and central wavelength. The output will be a Spectrum1D object.

    Parameters
    ----------
    resolution: float
        resolution R defined as lambda / delta lambda.
    central_wavelength: quantity
        the middle of the bandpass of interest.
    """
    parameters = []

    def __init__(self, resolution, central_wavelength):
        self.resolution = resolution
        self.central_wavelength = central_wavelength

    def __call__(self, spectrum):
        R = self.resolution
        Lambda = self.central_wavelength.value
        wavelength = spectrum.wavelength.value
        
        conversionfactor = 2 * np.sqrt(2 * np.log(2))
        deltax = np.mean(wavelength[1:] - wavelength[0:-1])
        FWHM = Lambda / R
        sigma = (FWHM / deltax) / conversionfactor
        
        flux = spectrum.flux

        convolved_flux = gaussian_filter1d(flux, sigma, axis = 0, order = 0)

        return Spectrum1D.from_array(
            spectrum.wavelength.value,
            convolved_flux,
            dispersion_unit = spectrum.wavelength.unit,
            unit = spectrum.unit)


class Interpolate(object):

    """
    This class can be called to do a interpolation on a given spectrum. You must initialize it with the observed spectrum. The output will be a Spectrum1D object.

    Parameters
    ----------
    observed: Spectrum1D object
        This is the observed spectrum which you want to interpolate your (model) spectrum to.
    fill_value: float
        This can be used to take model spectra with all Nan fluxes and change them into fluxes will all very high float values.
    
    Caution
    -------
    If you end up with an interpolated flux array with constant values, that is a likely result of how np.interp reacts when the wavelength ranges are not proper. Switching the units to be the same is a (partial) solution to this.
    """

    parameters = []

    def __init__(self, observed, fill_value=1e99):
        self.observed = observed
        self.fill_value = fill_value
            
    def __call__(self, spectrum):
        if self.observed.wavelength.unit != spectrum.wavelength.unit:
            log.warning('"observed wavelength" and "spectrum wavelength" do not share the same units ({0} vs {1}). The units of "spectrum wavelength" will be converted to the units of "observed wavelength".'.format(observed_wavelength.unit, wavelength.unit))
        else:
            pass
        
        wavelength, flux = spectrum.wavelength.to(u.Unit(self.observed.wavelength.unit)), spectrum.flux
        if (self.observed.wavelength[0] <= wavelength[0] <= self.observed.wavelength[-1]) or (self.observed.wavelength[0] <= wavelength[-1] <= self.observed.wavelength[-1]) is False:
            if (wavelength[0] <= self.observed.wavelength[0] <= wavelength[-1]) or (wavelength[0] <= self.observed.wavelength[-1] <= wavelength[-1]) is True:
                pass
            elif (wavelength[0] <= self.observed.wavelength[0] <= wavelength[-1]) or (wavelength[0] <= self.observed.wavelength[-1] <= wavelength[-1]) is False:
                raise ValueError('"observed wavelength" and "spectrum wavelength" do not overlap on any wavelength range. This needs to be resolved.')
            else:
                pass
        elif (self.observed.wavelength[0] <= wavelength[0] <= self.observed.wavelength[-1]) or (self.observed.wavelength[0] <= wavelength[-1] <= self.observed.wavelength[-1]) is True:
            log.warning('"spectrum wavelength" does not span the entire range of "observed wavelength". This needs to be checked.')
        else:
            pass

        interpolated_flux = np.interp(self.observed.wavelength.value, wavelength.value, flux)
        if np.isnan(interpolated_flux[0]):
            interpolated_flux = np.ones_like(interpolated_flux) * self.fill_value
        else:
            pass
        return Spectrum1D.from_array(self.observed.wavelength.value, interpolated_flux, dispersion_unit = self.observed.wavelength.unit, unit = self.observed.unit)


class CCM89Extinction(object):
    parameters = ['a_v', 'r_v']

    def __init__(self, a_v=0.0, r_v=3.1):
        self.a_v = a_v
        self.r_v = r_v

    def __call__(self, spectrum):

        from specutils import extinction

        extinction_factor = 10**(-0.4 * extinction.extinction_ccm89(
            spectrum.wavelength, a_v = self.a_v, r_v = self.r_v))


        return Spectrum1D.from_array(
            spectrum.wavelength.value,
            extinction_factor * spectrum.flux,
            dispersion_unit = spectrum.wavelength.unit, unit = spectrum.unit)



def observe(model, wgrid, slit, seeing, overresolve, offset=0.):
    """Convolve a model with a seeing profile, truncated by a slit, & pixelate

    Parameters
    ----------
    model: Table (or dict-like)
       Holding wavelengths and fluxes in columns 'w', 'flux'
    wgrid: array
       Wavelength grid to interpolate model on
    slit: float
       Size of the slit in wavelength units
    seeing: float
       FWHM of the seeing disk in wavelength units
    overresolve: int
       Factor by which detector pixels are overresolved in the wavelength grid
    offset: float, optional
       Offset of the star in the slit in wavelength units (default 0.)

    Returns
    -------
    Convolved model: Table
       Holding wavelength grid and interpolated, convolved fluxes
       in columns 'w', 'flux'
    """
    # make filter
    wgridres = np.min(np.abs(np.diff(wgrid)))
    filthalfsize = np.round(slit/2./wgridres)
    filtgrid = np.arange(-filthalfsize,filthalfsize+1)*wgridres
    # sigma ~ seeing-fwhm/sqrt(8*ln(2.))
    filtsig = seeing/np.sqrt(8.*np.log(2.))
    filt = np.exp(-0.5*((filtgrid - offset)/filtsig)**2)
    filt /= filt.sum()
    # convolve with pixel width
    filtextra = int((overresolve-1)/2+0.5)
    filt = np.hstack((np.zeros(filtextra), filt, np.zeros(filtextra)))
    filt = nd.convolve1d(filt, np.ones(overresolve)/overresolve)
    mint = np.interp(wgrid, model['w'], model['flux'])
    mconv = nd.convolve1d(mint, filt)
    return Table([wgrid, mconv], names = ('w','flux'), meta = {'filt': filt})
