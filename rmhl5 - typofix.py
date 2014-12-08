import numpy as np
import scipy as sp
from scipy import signal
from scipy.io import wavfile
from numpy import linalg as LA
import mpl_toolkits.mplot3d
from scipy.integrate import odeint
import pylab as plt



# Authors: Tammy Tran, Nuttida Rungratsameetaweemana, Brad Theilman

tau = 10.0  # time constant
p = 0.1 # internal connectivity

Ng = 1000.0 # Network Size

lambd = 1.2 # chaoticity level

train_dur = 500*1000 #60 seconds in ms

# noise in the firing rate
theta_state = 0.05
def zeta_state(t):
    return np.random.uniform(-theta_state, theta_state, Ng)

# exploration noise
# def zeta(t):
#     return np.random.uniform(low = -0.5, high = 0.5)

# decaying learning rate
eta_init = 0.0005
T = 20000.0 #s
def eta(t):
    return eta_init/(1 + ((t-2000)/T))
        # return eta_init

tau_avg = 5.0 #ms

# Heaviside
def heaviside(s, intermediate):
    return 1.0*(s > 0)
    

# Exponential Filter
tau_l = 50 #ms
def g(s):
    return sp.exp(-s/tau_l)*heaviside(s, 1)

# H smoothing functio
def h(s):
    return heaviside(s, 1) - heaviside(s - 100, 1)


# input streams
def u_on(t):
    return np.random.random_sample((sp.size(t),)) < 0.0005
def u_off(t):
    return np.random.random_sample((sp.size(t),)) < 0.0005
#def u_on_steam(t):
#def u_off_steam(t):
    
# target function
f_prev = 0
def f(uon, uoff):
    f_prev = 0
    uon_len = sp.size(uon)
    uoff_len = sp.size(uoff)
    if uon_len != uoff_len:
        return 0

    fret = sp.zeros(uon_len)
    for samp in sp.arange(1, uon_len):
        if uon[samp] == 1:
            f_prev = 0.5
        if uoff[samp] == 1:
            f_prev = -0.5
        fret[samp] = f_prev
    return fret


# Weight setup
sigma_p = np.sqrt(1/(p * Ng)) #why is there a square root?
sigma_w = np.sqrt(1/(Ng))
 
W_rec = sigma_p * np.random.randn(Ng, Ng) #weights in recurrent network
W_in_on = np.random.uniform(-1, 1, Ng)     #weights for input
W_in_off = np.random.uniform(-1, 1, Ng)
W_in_on[sp.random.random_sample(Ng) > 0.1] = 0
W_in_off[sp.random.random_sample(Ng) > 0.1] = 0 # sparseness for input weights

W_fb = np.random.uniform(-1, 1, Ng) #weights for feedback
W_rec[np.random.random_sample((Ng, Ng)) > p] = 0

w = np.random.randn(Ng)*sigma_w #weights for output
x = np.random.randn(Ng)*sigma_w #random initial x for network
w = np.zeros(Ng)
x = np.zeros(Ng)


svinit =  np.concatenate((x, w))
print(" Initialization Complete  ")

class RNN:

    def __init__(self, params):
        
        self.dt = params[0]
        self.training = params[1]
        self.f_target = params[2]
        self.uon = params[3]
        self.uoff = params[4]
        self.times = params[5]
        self.state = params[6]
        #self.xsave = np.zeros((samps, Ng))
        #self.wsave = np.zeros((samps, Ng))
        self.zsave = np.zeros(samps)
        self.rsave = np.zeros(samps)
        
        self.P_avg = 0
        self.z_avg = 0
        
    def dNeurons(self, statevec, t):
    
        # Extract relevant parameters from
        train = training > 0 and t > training and t < training + train_dur

        x_i = statevec[0:Ng]
        w_i = statevec[Ng:2*Ng]

        # generate noise patterns
        exp_noise_amp = 0.1
        if train:
        
            zeta = np.random.uniform(-exp_noise_amp, exp_noise_amp, 1)
        else:
            zeta = 0.0
            
        # Compute Firing Rates and feedback signals
        r_i = sp.tanh(x_i) + zeta_state(t)
        z_i = np.dot(w_i, r_i) + zeta
        
        # Compute next timestep depending on if training or not
        if train:
        
            dxidt = (-x_i + lambd * np.dot(W_rec, r_i) + np.dot(W_in_on, self.uon[t]) + np.dot(W_in_off, self.uoff[t]) + z_i*W_fb )/tau
            x_new = x_i + dxidt*dt
        
            P = -1.0*sp.power(z_i - self.f_target[t], 2)
            M = 1.0*(P > self.P_avg)
        
            dwdt = eta(t) * (z_i - self.z_avg) * M * r_i
            w_new = w_i + dwdt*dt

            self.P_avg = (1 - (dt/tau_avg)) * self.P_avg + (dt/tau_avg) * P
            self.z_avg = (1 - (dt/tau_avg)) * self.z_avg + (dt/tau_avg) * z_i
        
        else:
        
            dxidt = (-x_i + lambd * np.dot(W_rec, r_i) + np.dot(W_in_off, self.uoff[t]) +  np.dot(W_in_on, self.uon[t]) + z_i*W_fb)/tau
                    
            x_new = x_i + dxidt*dt
            dwdt = np.zeros(np.shape(w_i))
            w_new = w_i
            
        #weight change magnitude:
        dwmag = LA.norm(dwdt)
        rmag = LA.norm(r_i)
        if t%10000 == 0: print(str(t) + '    ' + str(z_i))
        self.zsave[t] = z_i
        self.rsave[t] = rmag
        return np.concatenate((x_new, w_new))
            
    def run_network(self):
        for indx, t in enumerate(self.times):
            #self.xsave[indx, :] = self.state[0:Ng]
            #self.wsave[indx, :] = self.state[Ng:2*Ng]
            self.state = self.dNeurons(self.state, t)
    
# Run the network
dt = 1
pre_train_dur = 2000
post_train_dur = 60*1000 # 10 seconds 
dur = pre_train_dur + train_dur + post_train_dur
times = sp.arange(0.0, dur, dt)
samps = np.size(times)

training = pre_train_dur

# generate input patterns
uon_t = 1*u_on(times)
uoff_t = 1*u_off(times)
f_raw = 1*f(uon_t, uoff_t)

# Generate filters
filt_t = sp.linspace(-200, 200, 401)
h_filt = h(filt_t)
g_filt = g(filt_t)

# compute filtered inputs
sigma_u = 1
uon_filt = sp.convolve(heaviside(sp.convolve(uon_t, h_filt, 'same'), 0), g_filt, 'same')/sigma_u
uoff_filt = sp.convolve(heaviside(sp.convolve(uoff_t, h_filt, 'same'), 0), g_filt, 'same')/sigma_u

uon_scale_factor = 0.4/uon_filt.max()
uoff_scale_factor = 0.4/uoff_filt.max()
uon_in = uon_filt * uon_scale_factor
uoff_in = uoff_filt * uoff_scale_factor

f_filt = sp.convolve(f_raw, g_filt, 'same')
f_scale_factor = 0.5/f_filt.max()
f_target = f_filt*f_scale_factor

params = (dt, training, f_target, uon_in, uoff_in, times, svinit)

maassnet = RNN(params)
maassnet.run_network()

# Determine performance
test_times = times[-post_train_dur:]
test_times = test_times.astype(int)
criterion = 0.5
perform = sum(1. * (abs(maassnet.zsave[test_times] - f_target[test_times]) < 0.5))/len(test_times)
print 'Performance: ', str(perform * 100), '%'

plt.figure()
plt.plot(maassnet.zsave)
plt.plot(maassnet.f_target)
plt.show()

# xsave = np.zeros((samps, Ng))
# wsave = np.zeros((samps, Ng))
# 
# # 
# # for indx, t in enumerate(times):
# #     xsave[indx, :] = svinit[0:Ng]
# #     wsave[indx, :] = svinit[Ng:2*Ng]
# #     svinit = dNeurons(svinit, t, params)

# rates = np.tanh(xsave)
# 
# rate_pre = rates - np.mean(rates, 0)
# rate_pre /= np.std(rate_pre, 0)
# rate_cov = np.cov(rates, rowvar=0)
# w, v = LA.eig(rate_cov)
# 
# prj1 = np.dot(rate_pre, v[:,0])
# prj2 = np.dot(rate_pre, v[:,1])
# prj3 = np.dot(rate_pre, v[:,2])
# 
# plt.figure().gca(projection='3d')
# plt.plot(prj1, prj2, prj3)
# 
# z = np.sum(wsave*rates, axis=1)
# plt.figure()
# plt.plot(times, z)
# 
# plt.show()
