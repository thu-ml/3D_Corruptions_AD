import numpy as np
import os
# import copy
# import math
# import pickle
# import argparse
# import open3d
# import struct
# import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.constants import speed_of_light as c     # in m/s

import functools
import numpy as np
import PyMieScatt as ps
import multiprocessing as mp

from typing import Tuple
from pathlib import Path
from scipy.special import gamma
from scipy.integrate import trapz
from tqdm.contrib.concurrent import process_map

PWD = str(Path(__file__).parent.resolve())


def multi_lisa(Rr: int, fixed_seed: bool, r_min: float, r_max: float, beam_divergence: float, min_diameter: float,
               refractive_index: float, range_accuracy: float, alpha: float, signal: str, density, diameters,
               p: Tuple[float, float, float, float],) -> Tuple[float, float, float, float, int, float]:

    x, y, z, i = p

    return monte_carlo_lisa(x=x, y=y, z=z, i=i, Rr=Rr, fixed_seed=fixed_seed, r_min=r_min, r_max=r_max, signal=signal,
                            beam_divergence=beam_divergence, min_diameter=min_diameter, range_accuracy=range_accuracy,
                            refractive_index=refractive_index, alpha=alpha, density=density, diameters=diameters)


def monte_carlo_lisa(x: float, y: float, z: float, i: float, Rr: int, fixed_seed: bool, r_min: float, r_max: float,
                     beam_divergence: float, min_diameter: float, refractive_index: float, range_accuracy: float,
                     alpha: float, signal: str, density, diameters) -> Tuple[float, float, float, float, int, float]:
    """
    For a single lidar return, performs a hybrid Monte-Carlo experiment

    :param
    x, y, z                 : coordinates of the point
    i                       : intensity [0 1]
    Rr                      : rain rate (mm/hr)

    :return
    x, y, z         : new coordinates of the noisy lidar point
    i_new           : new intensity
    label
    intensity_diff  : i - i_new
    """

    if fixed_seed:
        np.random.seed(666)

    r_new, i_new, label = None, None, None

    p_min = 0.9 * r_max ** (-2)                                     # min measurable power (arb units)

    beam_diameter = lambda d: 1e3 * np.tan(beam_divergence) * d     # beam diameter (mm) for a given distance (m)

    r = np.linalg.norm([x, y, z])

    if r > r_min:
        bvol = (np.pi / 3) * r * (1e-3 * beam_diameter(r) / 2) ** 2 # beam volume in m^3 (cone)
        n = density(Rr, min_diameter) * bvol                        # total number of particles in beam path
        n = np.int32(np.floor(n) + (np.random.rand() < n - int(n))) # convert to integer w/ probabilistic rounding
    else:
        n = 0

    # print(f'{n:3} snowflakes in {int(r):2}m range')

    particle_r_s = r * np.random.rand(n) ** (1 / 3)             # sample particle distances (Eq. 10)
    indx = np.where(particle_r_s > r_min)[0]                    # keep points where ranges larger than r_min
    particle_r_s = particle_r_s[indx]                           # this is typically always the case, but still sensible
    n = len(indx)                                               # new particle number

    p_hard = i * np.exp(-2 * alpha * r) / (r ** 2)              # power
    snr = p_hard / p_min                                        # signal noise ratio (Eq. 4)

    intensity_diff = 0

    if n > 0:

        particle_diameters = diameters(Rr, n, min_diameter)                 # sample n particle diameters (Eq. 12)
        fresnel = abs((refractive_index - 1) / (refractive_index + 1)) ** 2 # reflection at normal incidence (Eq. 11)

        # Calculate powers for all particles
        particle_p_s = fresnel * np.exp(-2 * alpha * particle_r_s) \
                               * np.minimum((particle_diameters / beam_diameter(particle_r_s)) ** 2, np.ones(n)) \
                               / (particle_r_s ** 2)
        # minimum in case the particle diameter is bigger than the beam diameter
        # and therefore fully occupies the LiDAR beam

        if signal == 'strongest':

            p_max_index = np.argmax(particle_p_s)                   # index of the max power
            p_particle = particle_p_s[p_max_index]
            r_particle = particle_r_s[p_max_index]
            particle_diameter = particle_diameters[p_max_index]

            if p_hard < p_min and p_particle < p_min:               # if all smaller than p_min, do nothing
                r_new = 0
                i_new = 0
                label = 0                                           # label as lost point

            elif p_hard < p_particle:                               # scatterer has larger power
                r_new = r_particle                                  # new range is scatterer range
                i_new = fresnel * np.exp(-2 * alpha * r_particle) \
                                * np.minimum((particle_diameter / beam_diameter(r_particle)) ** 2, 1)
                                                                    # new reflectance biased by scattering
                label = 2                                           # label as randomly scattered point

            else:                                                   # object return has larger power
                std = range_accuracy / np.sqrt(2 * snr)             # std of range uncertainty (Eq. 7)
                r_new = r + np.random.normal(0, std)                # range with uncertainty added
                i_new = i * np.exp(-2 * alpha * r)                  # new reflectance modified by scattering
                label = 1                                           # label as a non-scattered point

                intensity_diff = i - i_new

        elif signal == 'last':

            # if object power larger than p_min, then nothing is scattered
            if p_hard > p_min:

                std = range_accuracy / np.sqrt(2 * snr)             # std of range uncertainty
                r_new = r + np.random.normal(0, std)                # range with uncertainty added
                i_new = i * np.exp(-2 * alpha * r)                  # new reflectance modified by scattering
                label = 1                                           # label as a non-scattered point

                intensity_diff = i - i_new

            # otherwise find the furthest point above p_min
            else:

                inds = np.where(particle_p_s > p_min)[0]

                if len(inds) == 0:
                    r_new = 0
                    i_new = 0
                    label = 0                                               # label as lost point
                else:

                    particle_r_s = particle_r_s[inds]
                    p_last_index = np.argmax(particle_r_s)
                    r_particle = particle_r_s[p_last_index]
                    particle_diameter = particle_diameters[p_last_index]

                    r_new = r_particle                                      # new range is scatterer range
                    i_new = fresnel * np.exp(-2 * alpha * r_particle) \
                                    * np.minimum((particle_diameter / beam_diameter(r_particle)) ** 2, 1)
                                                                            # new reflectance biased by scattering
                    label = 2                                               # label as randomly scattered point

        else:

            print("Invalid lidar return mode")

    else:

        if p_hard < p_min:

            r_new = 0
            i_new = 0
            label = 0                                       # label as lost point

        else:

            std = range_accuracy / np.sqrt(2 * snr)         # std of range uncertainty
            r_new = r + np.random.normal(0, std)            # range with uncertainty added
            i_new = i * np.exp(-2 * alpha * r)              # new reflectance modified by scattering
            label = 1                                       # label as a non-scattered point

            intensity_diff = i - i_new

    # Angles are same
    if r > 0:
        phi = np.arctan2(y, x)      # angle in radians
        theta = np.arccos(z / r)    # angle in radians
    else:
        phi, theta = 0, 0

    # Update new x, y, z based on new range
    x = r_new * np.sin(theta) * np.cos(phi)
    y = r_new * np.sin(theta) * np.sin(phi)
    z = r_new * np.cos(theta)

    return x, y, z, i_new, label, intensity_diff


class LISA:

    def __init__(self, wavelength: float = 905, r_min: float = 0.9, r_max: float = 120, beam_divergence: float = 3e-3,
                 min_diameter: float = 0.05, range_accuracy: float = 0.09, signal: str = 'strongest',
                 mode: str = 'rain', show_progressbar: bool = False) -> None:
        """
        refractive_index    : refractive index of the droplets
        wavelength          : LiDAR wavelength (nm)
        rmin                : min lidar range (m)
        rmax                : max lidar range (m)
        beam_divergence     : beam divergence angle (rad)
        min_diameter        : min droplet diameter to be sampled (mm)
        range_accuracy      : range accuracy (m)
        saved_model         : use saved mie coefficients (bool)
        mode                : atmospheric type
        signal              : lidar return mode: "strongest" or "last"
        """

        self.r_min = r_min
        self.r_max = r_max
        self.signal = signal
        self.atm_model = mode
        self.wavelength = wavelength
        self.min_diameter = min_diameter
        self.range_accuracy = range_accuracy
        self.beam_divergence = beam_divergence
        self.show_progressbar = show_progressbar

        # diameter distribution function based on user input
        if mode == 'rain':
            self.refractive_index = 1.328  # refractive index of water
            self.Nd = self.marshall_palmer_Nd
            self.augment = self.monte_carlo_augment
            self.density = self.marshall_palmer_density
            self.diameters = self.marshall_palmer_sampling

        elif mode == 'gunn':
            self.refractive_index = 1.3031  # refractive index of ice
            self.Nd = self.marshall_gunn_Nd
            self.augment = self.monte_carlo_augment
            self.density = self.marshall_gunn_density
            self.diameters = self.marshall_gunn_sampling

        elif mode == 'sekhon':
            self.refractive_index = 1.3031  # refractive index of ice
            self.Nd = self.sekhon_srivastava_Nd
            self.augment = self.monte_carlo_augment
            self.density = self.sekhon_srivastava_density
            self.diameters = self.sekhon_srivastava_sampling

        elif mode == 'chu_hogg_fog':
            self.refractive_index = 1.328  # refractive index of water
            self.Nd = self.chu_hogg_Nd
            self.augment = self.average_augment

        elif mode == 'strong_advection_fog':
            self.refractive_index = 1.328  # refractive index of water
            self.Nd = self.strong_advection_fog_Nd
            self.augment = self.average_augment

        elif mode == 'moderate_advection_fog':
            self.refractive_index = 1.328  # refractive index of water
            self.Nd = self.moderate_advection_fog_Nd
            self.augment = self.average_augment

        elif mode == 'coast_haze':
            self.refractive_index = 1.328  # refractive index of water
            self.Nd = self.haze_coast_Nd
            self.augment = self.average_augment

        elif mode == 'continental_haze':
            self.refractive_index = 1.328  # refractive index of water
            self.Nd = self.haze_continental_Nd
            self.augment = self.average_augment

        elif mode == 'moderate_spray':
            self.refractive_index = 1.328  # refractive index of water
            self.Nd = self.moderate_spray_Nd
            self.augment = self.average_augment

        elif mode == 'strong_spray':
            self.refractive_index = 1.328  # refractive index of water
            self.Nd = self.strong_spray_Nd
            self.augment = self.average_augment

        elif mode == 'goodin et al.':
            self.refractive_index = 1.328  # refractive index of water
            self.Nd = self.goodin_Nd
            self.augment = self.goodin_augment

        try:
            dat = np.load(f'{PWD}/mie_{self.refractive_index}_λ_{self.wavelength}.npz')
            self.D = dat['D']
            self.qext = dat['qext']
            self.qback = dat['qback']
        except FileNotFoundError:
            # calculate Mie parameters
            print('Calculating Mie coefficients... \nThis might take a few minutes')
            self.D, self.qext, self.qback = self.calc_Mie_params()
            print('Mie calculation done...')


    def monte_carlo_augment(self, pc: np.ndarray, Rr: int, fixed_seed: bool = False) -> np.ndarray:
        """
        Augment clean pointcloud for a given rain rate
        :param
        pc : pointcloud (N,4) -> x, y, z, intensity
        Rr : rain rate (mm/hr)

        :return
        pc_new : new noisy point cloud (N,6) -> x, y, z, intensity, label                               , intensity_diff
                                                                    label 0 -> lost point
                                                                    label 1 -> randomly scattered point
                                                                    label 2 -> not-scattered
        """
        pc_new = np.zeros((pc.shape[0], pc.shape[1] + 2))

        point_list = list(map(tuple, pc[:, :4]))

        r_min = self.r_min
        r_max = self.r_max
        signal = self.signal
        density = self.density
        diameters = self.diameters
        min_diameter = self.min_diameter
        range_accuracy = self.range_accuracy
        beam_divergence = self.beam_divergence
        refractive_index = self.refractive_index

        curve = self.Nd(self.D, Rr)                     # density of the particles (m^-3)
        a = self.alpha(curve)

        if self.show_progressbar:

            pc_new[:, :] = process_map(functools.partial(multi_lisa, Rr, fixed_seed, r_min, r_max, beam_divergence,
                                                         min_diameter, refractive_index, range_accuracy, a, signal,
                                                         density, diameters), point_list, chunksize=1000)

        else:

            pool = mp.pool.ThreadPool(mp.cpu_count())

            pc_new[:, :] = pool.map(functools.partial(multi_lisa, Rr, fixed_seed, r_min, r_max, beam_divergence,
                                                      min_diameter, refractive_index, range_accuracy, a, signal,
                                                      density, diameters), point_list)

        return pc_new


    def average_augment(self, pc: np.ndarray, Rr: int = None, fixed_seed: bool = False) -> np.ndarray:

        if fixed_seed:
            np.random.seed(666)

        l, w = pc.shape
        pc_new = np.zeros((l, w + 2))                               # init new point cloud

        x, y, z, intensity = pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3]

        r_max = self.r_max                                          # max range (m)
        p_min = 0.9 * r_max ** (-2)                                 # min measurable power (arb units)
        r_min = self.r_min                                          # min lidar range (bistatic)

        Nd = self.Nd(self.D, Rr=Rr)                                 # density of rain droplets (m^-3)
        alpha = self.alpha(Nd)

        r = np.linalg.norm([x, y, z], axis=0)                       # range (m)
        indx = np.where(r > r_min)[0]                               # keep points where ranges larger than r_min

        p0 = np.zeros(l)                                            # init back reflected power
        std = np.zeros(l)                                           # init std of range uncertainty
        r_new = np.zeros(l)                                         # init new range

        p0[indx] = intensity[indx] * np.exp(-2 * alpha * r[indx]) / (r[indx] ** 2)  # calculate reflected power

        snr = p0 / p_min                                            # signal noise ratio
        indp = np.where(p0 > p_min)[0]                              # keep points where power is larger than p_min
        std[indp] = self.range_accuracy / np.sqrt(2 * snr[indp])    # calculate std of range uncertainty

        r_new[indp] = r[indp] + np.random.normal(0, std[indp])      # range with uncertainty added
        i_new = intensity * np.exp(-2 * alpha * r)                  # new reflectance modified by scattering

        # Init angles
        phi = np.zeros((l,))
        the = np.zeros((l,))

        phi[indx] = np.arctan2(y[indx], x[indx])                    # angle in radians
        the[indx] = np.arccos(z[indx] / r[indx])                    # angle in radians

        # Update new x,y,z based on new range
        pc_new[:, 0] = r_new * np.sin(the) * np.cos(phi)
        pc_new[:, 1] = r_new * np.sin(the) * np.sin(phi)
        pc_new[:, 2] = r_new * np.cos(the)
        pc_new[:, 3] = i_new
        pc_new[indp, 4] = 2
        pc_new[:, 5] = intensity - i_new

        return pc_new


    def goodin_augment(self, pc: np.ndarray, Rr: int, fixed_seed: bool = False) -> np.ndarray:
        """
        Lidar rain simulator from Goodin et al., 'Predicting the Influence of Rain on LIDAR in ADAS', electronics 2019

        :param
        pc : point cloud (N,4)
        Rr : rain rate in mm/hr

        :return
        pc_new : output point cloud (N,6)
        """
        if fixed_seed:
            np.random.seed(666)

        l, w = pc.shape
        pc_new = np.zeros((l, w + 2))

        x, y, z, i = pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3]

        r_max = self.r_max                                                  # max range (m)
        p_min = 0.9 * r_max ** (-2) / np.pi                                 # min measurable power (arb units)

        alpha = self.alpha(np.array(Rr))

        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)                               # range (m)
        indv = np.where(r > 0)[0]                                           # clean data might already have invalid pts
        p0 = np.zeros((l,))
        p0[indv] = i[indv] * np.exp(-2 * alpha * r[indv]) / (r[indv] ** 2)  # calculate reflected power

        r_new = np.zeros((l,))
        i_new = np.zeros((l,))

        indp = np.where(p0 > p_min)[0]                                      # points where power is greater than p_min
        i_new[indp] = i[indp] * np.exp(-2 * alpha * r[indp])                # reflectivity reduced by atten
        sig = 0.02 * r[indp] * (1 - np.exp(-Rr)) ** 2
        r_new[indp] = r[indp] + np.random.normal(0, sig)                    # new range with uncertainty

        # Init angles
        phi = np.zeros((l,))
        the = np.zeros((l,))

        phi[indp] = np.arctan2(y[indp], x[indp])                            # angle in radians
        the[indp] = np.arccos(z[indp] / r[indp])                            # angle in radians

        # Update new x,y,z based on new range
        pc_new[:, 0] = r_new * np.sin(the) * np.cos(phi)
        pc_new[:, 1] = r_new * np.sin(the) * np.sin(phi)
        pc_new[:, 2] = r_new * np.cos(the)
        pc_new[:, 3] = i_new
        pc_new[indp, 4] = 2
        pc_new[:, 5] = i - i_new

        return pc_new


    def calc_Mie_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate scattering efficiencies
        :return
        d     : Particle diameter (mm)
        qext  : Extinction efficiency
        qback : Backscattering efficiency

        """
        out = ps.MieQ_withDiameterRange(m = self.refractive_index, wavelength = self.wavelength, nd = 2000, logD = True,
                                        diameterRange = (1, 1e7)) # 1nm to 1cm

        d = out[0] * 1e-6
        qext = out[1]
        qback = out[6]

        # save for later use since this function takes long to run
        np.savez(f'{PWD}/mie_{self.refractive_index}_λ_{self.wavelength}.npz', D=d, qext=qext, qback=qback)

        return d, qext, qback


    def alpha(self, curve: np.ndarray) -> float:
        """
        Calculates extinction coefficient

        :param
        curve : particle size distribution, m^-3 mm^-1
        :return
        alpha : extinction coefficient
        """
        if curve.size == 1:
            alpha = 0.01 * curve ** 0.6
        else:
            alpha = 1e-6 * trapz(self.D ** 2 * self.qext * curve, self.D) * np.pi / 4

        return alpha # m^-1


    def beta(self, curve: np.ndarray) -> float:
        """
        Calculates the backscattering coefficient

        :param   
        curve : particle size distribution, m^-3 mm^-1
        :return
        beta  : backscattering coefficient
        """
        return 1e-6 * trapz(self.D ** 2 * self.qback * curve, self.D) * np.pi / 4  # m^-1


    # RAIN
    @staticmethod
    def goodin_Nd(D: np.ndarray, Rr: int) -> np.ndarray:
        return np.array(Rr)

    @staticmethod
    def marshall_palmer_Nd(D: np.ndarray, Rr: int) -> np.ndarray:
        """
        Marshall - Palmer rain model

        :param
        D  : rain droplet diameter (mm)
        Rr : rain rate (mm h^-1)

        :return
        number of rain droplets for a given diameter (m^-3 mm^-1)
        """
        return 8000 * np.exp(-4.1 * Rr ** (-0.21) * D)


    @staticmethod
    def marshall_palmer_density(Rr: int, dstart: float) -> float:
        """
        Integrated Marshall - Palmer Rain model

        :param
        Rr     : rain rate (mm h^-1)
        dstart : integral starting point for diameter (mm)

        :return
        rain droplet density (m^-3) for a given min diameter
        """
        Lambda = 4.1 * Rr ** (-0.21)

        return 8000 * np.exp(-Lambda * dstart) / Lambda


    @staticmethod
    def marshall_palmer_sampling(Rr: int, N: int, dstart: float) -> np.ndarray: # array of size N
        """
        Sample particle diameters from Marshall Palmer distribution

        :param
        Rr     : rain rate (mm/hr)
        N      : number of samples
        dstart : Starting diameter (min diameter sampled)

        :return
        diameters : diameter of the samples

        """
        Lambda = 4.1 * Rr ** (-0.21)
        r = np.random.rand(N)

        return -np.log(1 - r) / Lambda + dstart


    # SNOW
    @staticmethod
    def marshall_gunn_Nd(D: np.ndarray, Rr: int) -> np.ndarray:
        """
        Marshall Gunn (1958) snow model

        :param
        D  : snow diameter (mm)
        Rr : water equivalent rain rate (mm h^-1)

        :return
        number of snow particles for a given diameter (m^-3 mm^-1)
        """
        N0 = 7.6e3 * Rr ** (-0.87)
        Lambda = 2.55 * Rr ** (-0.48)

        return N0 * np.exp(-Lambda * D)


    @staticmethod
    def marshall_gunn_density(Rr: int, dstart: float) -> float:
        """
        Integrated Marshall Gunn (1958) snow model

        :param
        Rr     : rain rate (mm h^-1)
        dstart : integral starting point for diameter (mm)

        :return
        snow particle density (m^-3) for a given min diameter
        """
        N0 = 7.6e3 * Rr ** (-0.87)
        Lambda = 2.55 * Rr ** (-0.48)

        return N0 * np.exp(-Lambda * dstart) / Lambda


    @staticmethod
    def marshall_gunn_sampling(Rr: int, N: int, dstart: float) -> np.ndarray:   # array of size N
        """
        Sample particle diameters from Marshall Gunn (1958) distribution

        :param
        Rr     : rain rate (mm/hr)
        N      : number of samples
        dstart : Starting diameter (min diameter sampled)

        :return
        diameters : diameter of the samples

        """
        Lambda = 2.55 * Rr ** (-0.48)
        r = np.random.rand(N)

        return -np.log(1 - r) / Lambda + dstart


    @staticmethod
    def sekhon_srivastava_Nd(D: np.ndarray, Rr: int) -> np.ndarray:
        """
        Sekhon Srivastava (1970) snow model

        :param
        D  : snow diameter (mm)
        Rr : water equivalent rain rate (mm h^-1)

        :return
        number of snow particles for a given diameter (m^-3 mm^-1)
        """
        N0 = 5.0e3 * Rr ** (-0.94)
        Lambda = 2.29 * Rr ** (-0.45)

        return N0 * np.exp(-Lambda * D)


    @staticmethod
    def sekhon_srivastava_density(Rr: int, dstart: float) -> float:
        """
        Integrated Sekhon Srivastava (1970) snow model

        :param
        Rr     : rain rate (mm h^-1)
        dstart : integral starting point for diameter (mm)

        :return
        snow particle density (m^-3) for a given min diameter
        """
        N0 = 5.0e3 * Rr ** (-0.94)
        Lambda = 2.29 * Rr ** (-0.45)

        return N0 * np.exp(-Lambda * dstart) / Lambda


    @staticmethod
    def sekhon_srivastava_sampling(Rr: int, N: int, dstart: float) -> np.ndarray:  # array of size N
        """
        Sample particle diameters from Sekhon Srivastava (1970) distribution

        :param
        Rr     : rain rate (mm/hr)
        N      : number of samples
        dstart : Starting diameter (min diameter sampled)

        :return
        diameters : diameter of the samples

        """
        Lambda = 2.29 * Rr ** (-0.45)
        r = np.random.rand(N)

        return -np.log(1 - r) / Lambda + dstart


    # FOG
    @staticmethod
    def gamma_distribution_Nd(D: np.ndarray, rho: int, alpha: int, g: float, Rc: float) -> np.ndarray:
        """
        Gamma distribution model
        Note the parameters are NOT normalized to unitless values
        For example D^alpha term will have units Length^alpha
        It is therefore important to use exactly the same units for D as those
        cited in the paper by Rasshofer et al. and then perform unit conversion
        after an N(D) curve is generated

        D  : rain diameter
        Outputs number of rain droplets for a given diameter
        """
        b = alpha / (g * Rc ** g)

        Nd = g * rho * b ** ((alpha + 1) / g) * (D / 2) ** alpha * np.exp(-b * (D / 2) ** g) / gamma((alpha + 1) / g)

        return Nd


    # Coastal fog distribution
    # With given parameters, output has units cm^-3 um^-1 which is
    # then converted to m^-3 mm^-1 which is what alpha_beta() expects
    # so whole quantity is multiplied by (100 cm/m)^3 (1000 um/mm)
    def haze_coast_Nd(self, D: np.ndarray, Rr: int = None) -> np.ndarray:
        return 1e9 * self.gamma_distribution_Nd(D * 1e3, rho=100, alpha=1, g=0.5, Rc=0.05e-3)


    # Continental fog distribution
    def haze_continental_Nd(self, D: np.ndarray, Rr: int = None) -> np.ndarray:
        return 1e9 * self.gamma_distribution_Nd(D * 1e3, rho=100, alpha=2, g=0.5, Rc=0.07)


    # Strong advection fog
    def strong_advection_fog_Nd(self, D: np.ndarray, Rr: int = None) -> np.ndarray:
        return 1e9 * self.gamma_distribution_Nd(D * 1e3, rho=20, alpha=3, g=1., Rc=10)


    # Moderate advection fog
    def moderate_advection_fog_Nd(self, D: np.ndarray, Rr: int = None) -> np.ndarray:
        return 1e9 * self.gamma_distribution_Nd(D * 1e3, rho=20, alpha=3, g=1., Rc=8)


    # Strong spray
    def strong_spray_Nd(self, D: np.ndarray, Rr: int = None) -> np.ndarray:
        return 1e9 * self.gamma_distribution_Nd(D * 1e3, rho=100, alpha=6, g=1., Rc=4)


    # Moderate spray
    def moderate_spray_Nd(self, D: np.ndarray, Rr: int = None) -> np.ndarray:
        return 1e9 * self.gamma_distribution_Nd(D * 1e3, rho=100, alpha=6, g=1., Rc=2)


    # Chu/Hogg
    def chu_hogg_Nd(self, D: np.ndarray, Rr: int = None) -> np.ndarray:
        return 1e9 * self.gamma_distribution_Nd(D * 1e3, rho=20, alpha=2, g=0.5, Rc=1)