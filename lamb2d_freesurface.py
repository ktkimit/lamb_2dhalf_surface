import math
import numpy as np
import scipy.optimize
from scipy.integrate import quad
import matplotlib.pyplot as plt


class WaveSource(object):

    """Line load acting on the free surface of plane strain semi-infinite medium

    The location of this source is defined in Cartesian coordinates as x=0, z=0.
    """

    def evaluate(self, t):
        """Evaluate the function at time t

        :param t: time
        :returns: function values at time t

        """
        raise NotImplementedError("Child class should define method evaluate")

    def plotting(self, nt, tmax):
        """Plot the source function from time=0 to time=tmax

        :param nt: number of evaluation points
        :returns: TODO

        """
        raise NotImplementedError("Child class should define method plotting")

class Ricker(WaveSource):
    def __init__(self, a, freq, delay):
        """TODO: Docstring for __init__.

        :param freq: peak frequency
        :param dealy: time delay
        :returns: TODO

        """
        self.a = a
        self.freq = freq
        self.delay = delay
        
    def evaluate(self, t):
        x = np.pi*self.freq*(t - self.delay)
        x2 = x**2
        return self.a*(1.0 - 2.0*x2) * np.exp(-x2)

    def plotting(self, nt, tmax):
        t = np.linspace(0., tmax, num=nt)
        y = self.evaluate(t)

        plt.plot(t, y)
        plt.xlabel("Time")
        plt.show()

class Step3(WaveSource):
    def __init__(self, green, magnitude=1.0, timesteps=(0.05,0.1,0.15)):
        self.green = green
        self.magnitude = magnitude
        self.timesteps = timesteps

    def evaluate(self, t):
        if t <= self.timesteps[0]:
            return self.magnitude
        elif t <= self.timesteps[1]:
            return -2.0*self.magnitude
        elif t <= self.timesteps[2]:
            return self.magnitude
        else:
            return 0.0

    def evaluate_convolution(self, x, t, tt):
        """
        Evaluate F(tt)*green(t-tt)

        :param x float: receiver's location
        :param t float: time
        :param tt float: integration variable
        """
        f = self.evaluate(tt)
        if f == 0.0:
            return (0.0, 0.0)
        
        bt = t - tt
        u = self.green.evaluate(x, self.green.t2tau(x, bt))

        return tuple(f*val for val in u)

    def integration_convolution(self, x, t):
        """
        Integrate F(tt)*green(t-tt) dtt from 0 to t

        :param x float: receiver's location
        :param t float: time
        """
        if (self.green.t2tau(x,t)<1.0 and self.green.t2tau(x,t)<self.green.k):
            return (0.0, 0.0)

        tt_upper = self.timesteps[2]

        t_one = self.green.tau2t(x, 1.0)
        t_k = self.green.tau2t(x, self.green.k)
        t_r = self.green.tau2t(x, self.green.zetar)

        tt_one = t - t_one
        tt_k = t - t_k
        tt_r = t - t_r

        points = list(self.timesteps)
        points.append(tt_one)
        points.append(tt_k)
        points.append(tt_r)
        points.sort()

        points = np.array(points)
        points = points[ (points>=0.0) * (points<=tt_upper) ]

        fu = lambda tt: self.evaluate_convolution(x, t, tt)[0]
        fw = lambda tt: self.evaluate_convolution(x, t, tt)[1]

        u, u_err = quad(fu, 0.0, tt_upper, points=points)
        if tt_r >= 0.0 and tt_r <= tt_upper:
            u += self.evaluate(tt_r)*self.green.evaluate(x, self.green.zetar)[0]

            fw_nosingular = lambda tt: fw(tt)*(tt - tt_r)
            w1, w1_err = quad(fw_nosingular, 0.0, tt_k, weight='cauchy', wvar=tt_r)
            
            points = points[ (points>=tt_k) * (points<=tt_upper) ]
            w2, w2_err = quad(fw, tt_k, tt_upper, points=points)

            w = w1 + w2
        else:
            w, w_err = quad(fw, 0.0, tt_upper, points=points)

        return (u, w)

    def plot_response(self, x, tmax, nt):
        t = np.linspace(0., tmax, num=nt)
        uhist = np.zeros(t.size)
        whist = np.zeros(t.size)
        for i, time in enumerate(t):
            (uhist[i], whist[i]) = self.integration_convolution(x, time)

        fig, axs = plt.subplots(2, sharex=True)
        axs[0].plot(t, uhist)
        axs[1].plot(t, whist)

        axs[0].set_ylabel(r'$u_x$')
        axs[1].set_ylabel(r'$u_z$')
        axs[1].set_xlabel(r'$time$')

        plt.margins(x=0)
        plt.show()

    def plot_convolution(self, x, t, ntt):
        """
        Plot the colvolution of F(tt)*green(t-tt)

        :param t float: time
        :param ntt int: number of evaluation points
        :param x float: receiver's location
        """
        tt = np.linspace(0., t, num=ntt)
        u = np.zeros(tt.size)
        w = np.zeros(tt.size)
        for i, z in enumerate(tt):
            (u[i], w[i]) = self.evaluate_convolution(x, t, z)

        fig, axs = plt.subplots(2, sharex=True)
        axs[0].plot(tt, u)
        axs[1].plot(tt, w)

        axs[0].set_ylabel(r'$u_x$')
        axs[1].set_ylabel(r'$u_z$')
        axs[1].set_xlabel(r'$\tt$')

        plt.margins(x=0)
        plt.show()
       

class GreenfunctionFreesurface(object):

    """Green's function on the free surface for a delta source located on x=0, z=0

    Reference:
    The Theory of Elastic Waves and Waveguides, J. Miklowitz

    """

    def __init__(self, rho, lmbda, mu):
        """TODO: to be defined.

        :param rho: density
        :param lmbda: Lamé's first parameter 
        :param mu: Lamé's second parameter 

        """
        self.cs = math.sqrt( mu / rho )
        self.cd = math.sqrt( (lmbda + 2.*mu) / rho )
        self.k = self.cd / self.cs
        self.k2 = self.k**2
        self.kr2 = self._solve_rayleigh_equation()
        self.zetar = self.k / math.sqrt(self.kr2)

        self.coef = self.cs / (math.pi*mu)

    def t2tau(self, x, t):
        """

        :param x: location of a receiver on the freesurface
        :param t: time
        :returns: tau

        """
        assert x > 0.0

        tau = self.cd*t / x
        return tau

    def tau2t(self, x, tau):
        assert x > 0.0

        t = tau*x / self.cd
        return t

    def evaluate(self, x, tau):
        """TODO: Docstring for eval_nondim.

        :param x: location of a receiver on the freesurface
        :param t: tau
        :returns: displacements in x an z directions, [ux, uz]

        """
        assert x > 0.0

        ux = 0.
        uz = 0.

        if tau < 1.0:
            return (ux, uz)

        tau2 = tau**2
        k3 = self.k2*self.k
        coef = self.coef / x
        if tau >= self.k:
            S = self.k2 - 2.*tau2
            T = np.sqrt(tau2 - 1.)
            V = np.sqrt(tau2 - self.k2)
            R2 = S**2 - 4.*tau2*T*V

            uz = k3*T/R2
            uz *= - coef

            # ux should be in fact infinite value
            if tau == self.zetar:
                G = 8.*(self.k2 - 1.) - 4.*self.k2*self.kr2**2 + self.k2*self.kr2**3
                ux = - 2.*k3 * math.pi*(2. - self.kr2)**3 / (8.*G)
                ux *= coef

            return (ux, uz)
        else:
            S = self.k2 - 2.*tau2
            T = np.sqrt(tau2 - 1.)
            U = np.sqrt(self.k2 - tau2)
            R1 = S**4 + (4.*tau2*T*U)**2

            ux = 2.*k3 * tau*S*T*U / R1
            uz = k3 * S**2*T / R1

            ux *= coef
            uz *= -coef

            return (ux, uz)

    def evaluate_nosingular(self, x, tau):
        (ux, uz) = self.evaluate(x, tau)
        uz *= (tau - self.zetar)
        return (ux, uz)
       
    def plotting(self, x, ntau, taumin, taumax):
        tau = np.linspace(taumin, taumax, num=ntau)

        ux = np.zeros(ntau)
        uz = np.zeros(ntau)

        for i in range(ntau):
            tt = tau[i]
            (ux[i], uz[i]) = self.evaluate(x, tt)

        fig, axs = plt.subplots(2, sharex=True)
        axs[0].plot(tau, ux)
        axs[1].plot(tau, uz)

        axs[0].set_ylabel(r'$u_x$')
        axs[1].set_ylabel(r'$u_z$')
        axs[1].set_xlabel(r'$\tau$')

        plt.margins(x=0)
        plt.show()

    def _solve_rayleigh_equation(self):
        """
        Solve the Rayleigh characteriatic equation to find kr.
        kr = cr / cs where cr is the Rayleigh wave velocity.

        :returns: kr**2

        """
        f = lambda x : x**3 - 8.*x**2 + 8.*x*(3. - 2. / self.k2) \
                - 16.*(1. - 1. / self.k2)
        y = scipy.optimize.brentq(f, 0., 1.)
        return y
        
        
# Young's modulus 
young = 1.877303655819164e10
# Poisson's ratio 
nu = 0.250008468532307
# density
rho = 2.2e3

lmbda = young*nu / ((1. + nu) * (1. - 2.*nu))
mu = young / (2.*(1.+nu))

green = GreenfunctionFreesurface(rho, lmbda, mu)
stepf = Step3(green, magnitude=1e6)

