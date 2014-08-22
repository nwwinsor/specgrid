from scipy import optimize
import numpy as np
from collections import OrderedDict


#A few notes on using the 'basinhopping' fitting routine. The way it is set up is as follows. You need to initial the classes RandomDisplacementBounds and Print_Function, as well as define "bounds".
#An example for my MARCS grid, fitting for four parameters (in the order of: log(g), vrad, Teff, Fe/H):
#xmin = [-0.5, -2000., 2500.0, -5.0]
#xmax = [5.5, 2000., 8000.0,  1.0]
#bounds = [(low, high) for low, high in zip(xmin, xmax)]
#take_step = RandomDisplacementBounds(xmin, xmax)
#print_fun = Print_Function()

class RandomDisplacementBounds(object):

    """
    This is a class which is used to give the fitting routine 'basinhopping' bounds and to adjust the stepsize for each variable of 'basinhopping'.

    Parameters
    ----------
    xmin: array
        This is an array which contains the minimum values of each parameter (i.e. Teff, Fe/H, log(g), vrad, etc.) for the desired grid bounds.
    xmax: array
        This is an array which contains the maximum values of each parameter (i.e. Teff, Fe/H, log(g), vrad, etc.) for the desired grid bounds.
    parameters: 4
        This tells the Class how many parameters you are fitting for. It is useful for when you want to change the number of parameters you wish to fit.
    LOGG: True or False
        This tells the Class if you are fitting for log(g). If this is set as True, then you are saying that you are fitting for this parameter
    VRAD: True or False
        This tells the Class if you are fitting for vrad. If this is set as True, then you are saying that you are fitting for this parameter
    TEFF: True or False
        This tells the Class if you are fitting for Teff. If this is set as True, then you are saying that you are fitting for this parameter
    FEH: True or False
        This tells the Class if you are fitting for Fe/H. If this is set as True, then you are saying that you are fitting for this parameter
    stepsizeG: float
        This sets the stepsize for the variable log(g) which 'basinhopping' will use. This should be set to be equal to the minimum distance between local minima for this variable.
    stepsizeV: float
        This sets the stepsize for the variable vrad which 'basinhopping' will use. This should be set to be equal to the minimum distance between local minima for this variable.
    stepsizeT: float
        This sets the stepsize for the variable Teff which 'basinhopping' will use. This should be set to be equal to the minimum distance between local minima for this variable.
    stepsizeZ: float
        This sets the stepsize for the variable Fe/H which 'basinhopping' will use. This should be set to be equal to the minimum distance between local minima for this variable.

    Caution
    --------
    This class is only generalized for four fit variables so far (Teff, log(g), Fe/H and vrad). It will have to be adjusted if you wish to fit other variables as well.
    If you wish to not fit for any of the parameters: Teff, log(g) or Fe/H, then you must set whichever parameters that you do not want to fit to some value using ModelStar.eval(). This must be done before you run the fitter. For example, if you wish to set Teff as 5000.0 K and log(g) as 3.0, then you would do: ModelStar.eval(teff = 5000.0, logg = 3.0).
    
    Note
    ----
    Initialize this class by calling it: "take_step"
    """
    def __init__(self, xmin, xmax, parameters = 4, LOGG = True, VRAD = True, TEFF = True, FEH = True, stepsizeG=1.0, stepsizeV = 200.0, stepsizeT=1000, stepsizeZ = 1.0):
        self.xmin = xmin
        self.xmax = xmax
        self.parameters = parameters
        self.LOGG = LOGG
        self.VRAD = VRAD
        self.TEFF = TEFF
        self.FEH = FEH
        if self.LOGG is True:
            self.stepsizeG = stepsizeG
        else:
            self.stepsizeG = 0.0
        if self.VRAD is True:
            self.stepsizeV = stepsizeV
        else:
            self.stepsizeV = 0.0        
        if self.TEFF is True:
            self.stepsizeT = stepsizeT
        else:
            self.stepsizeT = 0.0
        if self.FEH is True:
            self.stepsizeZ = stepsizeZ
        else:
            self.stepsizeZ = 0.0

    def __call__(self, x):

        while True:
            if self.parameters == 4:
                xG = x[0]
                xV = x[1]
                xT = x[2]
                xZ = x[3]
                xG_new = xG + np.random.uniform(-self.stepsizeG, self.stepsizeG, np.shape(xG))
                xV_new = xV + np.random.uniform(-self.stepsizeV, self.stepsizeV, np.shape(xV))
                xT_new = xT + np.random.uniform(-self.stepsizeT, self.stepsizeT, np.shape(xT))
                xZ_new = xZ + np.random.uniform(-self.stepsizeZ, self.stepsizeZ, np.shape(xZ))
                xnew = np.array([xG_new, xV_new, xT_new, xZ_new])
            
                if np.all(xnew < xmax) and np.all(xnew > xmin):
                    break
                else:
                    xnew = x

            elif self.parameters == 3 and self.FEH is not True:
                xG = x[0]
                xV = x[1]
                xT = x[2]
                xG_new = xG + np.random.uniform(-self.stepsizeG, self.stepsizeG, np.shape(xG))
                xV_new = xV + np.random.uniform(-self.stepsizeV, self.stepsizeV, np.shape(xV))
                xT_new = xT + np.random.uniform(-self.stepsizeT, self.stepsizeT, np.shape(xT))
                xnew = np.array([xG_new, xV_new, xT_new])
            
                if np.all(xnew < xmax) and np.all(xnew > xmin):
                    break
                else:
                    xnew = x

            elif self.parameters == 3 and self.TEFF is not True:
                xG = x[0]
                xV = x[1]
                xZ = x[2]
                xG_new = xG + np.random.uniform(-self.stepsizeG, self.stepsizeG, np.shape(xG))
                xV_new = xV + np.random.uniform(-self.stepsizeV, self.stepsizeV, np.shape(xV))
                xZ_new = xZ + np.random.uniform(-self.stepsizeZ, self.stepsizeZ, np.shape(xZ))
                xnew = np.array([xG_new, xV_new, xZ_new])
            
                if np.all(xnew < xmax) and np.all(xnew > xmin):
                    break
                else:
                    xnew = x

            elif self.parameters == 3 and self.LOGG is not True:
                xV = x[0]
                xT = x[1]
                xZ = x[2]
                xV_new = xV + np.random.uniform(-self.stepsizeV, self.stepsizeV, np.shape(xV))
                xT_new = xT + np.random.uniform(-self.stepsizeT, self.stepsizeT, np.shape(xT))
                xZ_new = xZ + np.random.uniform(-self.stepsizeZ, self.stepsizeZ, np.shape(xZ))
                xnew = np.array([xV_new, xT_new, xZ_new])
            
                if np.all(xnew < xmax) and np.all(xnew > xmin):
                    break
                else:
                    xnew = x

            elif self.parameters == 3 and self.VRAD is not True:
                xG = x[0]
                xT = x[1]
                xZ = x[2]
                xG_new = xG + np.random.uniform(-self.stepsizeG, self.stepsizeG, np.shape(xG))
                xT_new = xT + np.random.uniform(-self.stepsizeT, self.stepsizeT, np.shape(xT))
                xZ_new = xZ + np.random.uniform(-self.stepsizeZ, self.stepsizeZ, np.shape(xZ))
                xnew = np.array([xG_new, xT_new, xZ_new])
            
                if np.all(xnew < xmax) and np.all(xnew > xmin):
                    break
                else:
                    xnew = x

            elif self.parameters == 2 and self.FEH is not True and self.TEFF is not True:
                xG = x[0]
                xV = x[1]

                xG_new = xG + np.random.uniform(-self.stepsizeG, self.stepsizeG, np.shape(xG))
                xV_new = xV + np.random.uniform(-self.stepsizeV, self.stepsizeV, np.shape(xV))

                xnew = np.array([xG_new, xV_new])
            
                if np.all(xnew < xmax) and np.all(xnew > xmin):
                    break
                else:
                    xnew = x

            elif self.parameters == 2 and self.FEH is not True and self.VRAD is not True:
                xG = x[0]
                xT = x[1]

                xG_new = xG + np.random.uniform(-self.stepsizeG, self.stepsizeG, np.shape(xG))
                xT_new = xT + np.random.uniform(-self.stepsizeT, self.stepsizeT, np.shape(xT))

                xnew = np.array([xG_new, xT_new])
            
                if np.all(xnew < xmax) and np.all(xnew > xmin):
                    break
                else:
                    xnew = x

            elif self.parameters == 2 and self.FEH is not True and self.LOGG is not True:
                xV = x[0]
                xT = x[1]

                xV_new = xV + np.random.uniform(-self.stepsizeV, self.stepsizeV, np.shape(xV))
                xT_new = xV + np.random.uniform(-self.stepsizeT, self.stepsizeT, np.shape(xT))

                xnew = np.array([xV_new, xT_new])
            
                if np.all(xnew < xmax) and np.all(xnew > xmin):
                    break
                else:
                    xnew = x

            elif self.parameters == 2 and self.TEFF is not True and self.VRAD is not True:
                xG = x[0]
                xZ = x[1]

                xG_new = xG + np.random.uniform(-self.stepsizeG, self.stepsizeG, np.shape(xG))
                xZ_new = xZ + np.random.uniform(-self.stepsizeZ, self.stepsizeZ, np.shape(xZ))

                xnew = np.array([xG_new, xZ_new])
            
                if np.all(xnew < xmax) and np.all(xnew > xmin):
                    break
                else:
                    xnew = x

            elif self.parameters == 2 and self.TEFF is not True and self.LOGG is not True:
                xV = x[0]
                xZ = x[1]

                xV_new = xV + np.random.uniform(-self.stepsizeV, self.stepsizeV, np.shape(xV))
                xZ_new = xZ + np.random.uniform(-self.stepsizeZ, self.stepsizeZ, np.shape(xZ))

                xnew = np.array([xV_new, xZ_new])
            
                if np.all(xnew < xmax) and np.all(xnew > xmin):
                    break
                else:
                    xnew = x

            elif self.parameters == 2 and self.VRAD is not True and self.LOGG is not True:
                xT = x[0]
                xZ = x[1]

                xT_new = xT + np.random.uniform(-self.stepsizeT, self.stepsizeT, np.shape(xT))
                xZ_new = xZ + np.random.uniform(-self.stepsizeZ, self.stepsizeZ, np.shape(xZ))

                xnew = np.array([xT_new, xZ_new])
            
                if np.all(xnew < xmax) and np.all(xnew > xmin):
                    break
                else:
                    xnew = x

            elif self.parameters == 1 and self.LOGG is True:
                xG = x[0]

                xG_new = xG + np.random.uniform(-self.stepsizeG, self.stepsizeG, np.shape(xG))

                xnew = np.array([xG_new])
            
                if np.all(xnew < xmax) and np.all(xnew > xmin):
                    break
                else:
                    xnew = x

            elif self.parameters == 1 and self.VRAD is True:
                xV = x[0]

                xV_new = xG + np.random.uniform(-self.stepsizeV, self.stepsizeV, np.shape(xV))

                xnew = np.array([xV_new])
            
                if np.all(xnew < xmax) and np.all(xnew > xmin):
                    break
                else:
                    xnew = x

            elif self.parameters == 1 and self.TEFF is True:
                xT = x[0]

                xT_new = xT + np.random.uniform(-self.stepsizeT, self.stepsizeT, np.shape(xT))

                xnew = np.array([xT_new])
            
                if np.all(xnew < xmax) and np.all(xnew > xmin):
                    break
                else:
                    xnew = x

            elif self.parameters == 1 and self.FEH is True:
                xZ = x[0]

                xZ_new = xZ + np.random.uniform(-self.stepsizeZ, self.stepsizeZ, np.shape(xZ))

                xnew = np.array([xZ_new])
            
                if np.all(xnew < xmax) and np.all(xnew > xmin):
                    break
                else:
                    xnew = x
            
            else:
                print "Check MyBounds Class; the fitter will not work proerly if you are receiving this message. Either there is a bug in the code, or there is a mismatch of input for parameters and the Truth values given. In example, if parameters = 3, then only 3 of the parameters should have a truth value of True."
                xnew = x
                break
        return xnew


class Print_Function(object):

    """
    This is a class which is used to tell the fitting routine 'basinhopping' to print the values of the parameters tested at each 'minima'.

    Parameters
    ----------
    print_values: True or False
        This tells the class if you want to print the values tested or not; if set as True it will print. Printing the values is useful for diagnostic purposes. The default setting is True.
    parameters: 4
        This tells the Class how many parameters you are fitting for. It is useful for when you want to change the number of parameters you wish to fit.
    LOGG: True or False
        This tells the Class if you are fitting for log(g). If this is set as True, then you are saying that you are fitting for this parameter.
    VRAD: True or False
        This tells the Class if you are fitting for vrad. If this is set as True, then you are saying that you are fitting for this parameter.
    TEFF: True or False
        This tells the Class if you are fitting for Teff. If this is set as True, then you are saying that you are fitting for this parameter.
    FEH: True or False
        This tells the Class if you are fitting for Fe/H. If this is set as True, then you are saying that you are fitting for this parameter.
    

    Caution
    --------
    This class is only generalized for four fit variables so far (Teff, log(g), Fe/H and vrad). It will have to be adjusted if you wish to use it to print the parameters tested while fitting different variables. Otherwise, it can still be used to print the function value and if it was accepted or not.
    
    Note
    ----
    Initialize this class by calling it: "print_fun".
    """

    def __init__(self, print_values = True, parameters = 4, LOGG = True, VRAD = True, TEFF = True, FEH = True):
        self.print_values = print_values
        self.parameters = parameters
        self.LOGG = LOGG
        self.VRAD = VRAD
        self.TEFF = TEFF
        self.FEH = FEH
    
    def __call__(self, x, f, accepted):
        if self.print_values is True:
            if self.parameters == 4:
                Minima = "at minima (logg = {0}, vrad = {1}, teff = {2}, feh = {3}),".format(x[0], x[1], x[2], x[3])
            elif self.parameters == 3 and self.FEH is not True:
                Minima = "at minima (logg = {0}, vrad = {1}, teff = {2}),".format(x[0], x[1], x[2])
            elif self.parameters == 3 and self.TEFF is not True:
                Minima = "at minima (logg = {0}, vrad = {1}, feh = {2}),".format(x[0], x[1], x[2])
            elif self.parameters == 3 and self.VRAD is not True:
                Minima = "at minima (logg = {0}, teff = {1}, feh = {2}),".format(x[0], x[1], x[2])
            elif self.parameters == 3 and self.LOGG is not True:
                Minima = "at minima (vrad = {0}, teff = {1}, feh = {2}),".format(x[0], x[1], x[2])
            elif self.parameters == 2 and self.FEH is not True and self.TEFF is not True:
                Minima = "at minima (logg = {0}, vrad = {1}),".format(x[0], x[1])
            elif self.parameters == 2 and self.FEH is not True and self.VRAD is not True:
                Minima = "at minima (logg = {0}, teff = {1}),".format(x[0], x[1])
            elif self.parameters == 2 and self.FEH is not True and self.LOGG is not True:
                Minima = "at minima (vrad = {0}, teff = {1}),".format(x[0], x[1])
            elif self.parameters == 2 and self.TEFF is not True and self.VRAD is not True:
                Minima = "at minima (logg = {0}, feh = {1}),".format(x[0], x[1])
            elif self.parameters == 2 and self.TEFF is not True and self.LOGG is not True:
                Minima = "at minima (vrad = {0}, feh = {1}),".format(x[0], x[1])
            elif self.parameters == 2 and self.VRAD is not True and self.LOGG is not True:
                Minima = "at minima (teff = {0}, feh = {1}),".format(x[0], x[1])
            elif self.parameters == 1 and self.LOGG is True:
                Minima = "at minima (logg = {0}),".format(x[0])
            elif self.parameters == 1 and self.VRAD is True:
                Minima = "at minima (vrad = {0}),".format(x[0])
            elif self.parameters == 1 and self.TEFF is True:
                Minima = "at minima (teff = {0}),".format(x[0])
            elif self.parameters == 1 and self.FEH is True:
                Minima = "at minima (feh = {0}),".format(x[0])
            else:
                Minima = ""
                print "Check Print Function Class. Either there is a bug in the code, or the use of __init__ has an input mismatch for parameters and the Truth values given. In example, if parameters = 3, then only 3 of the parameters should have a truth value of True."
            print Minima + " f = {0}, accepted = {1}".format(f, int(accepted))
        else:
            pass







def fit_spectrum_test(spectrum, guess, model_star, fitter='leastsq', bounds = None, take_step = None, print_fun = None, fill_value=1e99, valid_slice=slice(None)):
    """
    This function is used for fitting model spectral to an observed spectra.

    Parameters
    ----------
    spectrum: Spectrum1D object
        This is the observed spectrum
    guess: dictionary
        These is a dictionary of the initial guess values for the parameters being fit. 
    model_star: ModelStar class object
        This is the grid of model spectra (possibly with 'plugins' applied to them) being fit to the observed spectra.
    bounds: list (like: [(minlogg, maxlogg), (minteff, maxteff), (minfeh, maxfeh)]
        This is used for the basinhopping fitting routine. It acts as bounds for the grid. Order is important.
    take_step: RandomDisplacementBounds class object
        This is used for the basinhopping fitting routine. It gives basinhopping bounds as well as stepsizes for each variable.
    print_fun: Print_Function class object
        This is used for the basinhopping fitting routine. It tells basinhopping to print out each minimum found while fitting.
    fitter: string
        This is the desired fitting method. The two built-in fitters include: leastsq and basinhopping
    fill_value: float
        This is used to avoid issues with "Nan" values
    valid_slice: ???
        ???
    """
    if getattr(spectrum, 'uncertainty', None) is not None:
        uncertainty = spectrum.uncertainty.array
    else:
        uncertainty = np.ones_like(spectrum.flux)


    def spectral_model_fit(pars):
        pardict = OrderedDict()
        for key, par in zip(guess.keys(), pars):
            pardict[key] = par

        model = model_star.eval(**pardict)

        if getattr(spectrum, 'uncertainty', None) is not None:
            uncertainty = spectrum.uncertainty.array
        else:
            uncertainty = np.ones_like(spectrum.flux)

        if np.isnan(model.flux[0]):
            model_flux = np.ones_like(model.flux) * fill_value
        else:
            model_flux = model.flux

        quality = (((spectrum.flux - model_flux) / uncertainty)[valid_slice])**2
        #For 'basinhopping' (and other routines) the minimized function is reduced chi squared.
        return quality if fitter == 'leastsq' else quality.sum()/(spectrum.shape[0]-len(guess.keys()))


    if fitter == 'leastsq':
        fit = optimize.leastsq(spectral_model_fit, np.array(guess.values()),
                               full_output=True)
        

        stellar_params = OrderedDict((key, par)
                                     for key, par in zip(guess.keys(), fit[0]))

        function_value = 'Not Applicable for leastsq fitter'
        
        if fit[1] is not None:
            stellar_params_uncertainty = OrderedDict(
                (key, np.sqrt(par)) for key, par in
                zip(guess.keys(), np.diag(fit[1])))

        else:
            stellar_params_uncertainty = OrderedDict((key, None)
                                                     for key in guess.keys())
    elif fitter == 'basinhopping':
        #Note: Could use other 'methods' for minimizer_kwargs (i.e. BFGS)
        minimizer_kwargs = dict(method = 'L-BFGS-B', bounds = bounds)
        #Note: 'niter' could be adjusted to make the fitting process faster. This needs to be tested, to see which value of 'niter' will cause the fit parameters to always converge to the same result.
        fit = optimize.basinhopping(spectral_model_fit, np.array(guess.values()), niter = 100, interval = 50, minimizer_kwargs = minimizer_kwargs, take_step = take_step, callback = print_fun)
        
        stellar_params = OrderedDict(
            (key, par) for key, par in zip(guess.keys(), fit['x']))
        
        function_value = fit['fun']
        
        stellar_params_uncertainty = OrderedDict(
            (key, None) for key, par in zip(guess.keys(), fit['x']))
    
    else:
        fit = optimize.minimize(spectral_model_fit, np.array(guess.values()),
                                method=fitter)
        stellar_params = OrderedDict(
            (key, par) for key, par in zip(guess.keys(), fit['x']))
        function_value = 'Not Applicable here'
        stellar_params_uncertainty = OrderedDict(
            (key, None) for key, par in zip(guess.keys(), fit['x']))

    return stellar_params, stellar_params_uncertainty, fit, function_value if fitter = 'basinhopping' else stellar_params, stellar_params_uncertainty, fit
