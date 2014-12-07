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

# noise in the firing rate
theta_state = 0.05
def zeta_state(t):
    return np.random.uniform(-theta_state, theta_state, Ng)

# decaying learning rate
eta_init = 0.0005
T = 20000.0 #s
def eta(t):
    return eta_init/(1 + ((t-pre_train_dur)/T))
        # return eta_init

tau_avg = 5.0 #ms

# Weight setup
sigma_p = np.sqrt(1/(p * Ng)) 
 
W_rec = sigma_p * np.random.randn(Ng, Ng) #weights in recurrent network
W_in_left = np.random.uniform(-1, 1, Ng)     #weights for input
W_in_right = np.random.uniform(-1, 1, Ng)
W_in_left[sp.random.random_sample(Ng) > 0.1] = 0 # sparseness for input weights
W_in_right[sp.random.random_sample(Ng) > 0.1] = 0 

W_fb = np.random.uniform(-1, 1, Ng) #weights for feedback
W_rec[np.random.random_sample((Ng, Ng)) > p] = 0 #recurrent network has sparseness

w = np.zeros(Ng) #weights for output
x = np.zeros(Ng) #initial x for network

svinit =  np.concatenate((x, w))
print(" Initialization Complete  ")

class RNN:

    def __init__(self, params):
        
        self.dt = params[0]
        self.training = params[1]
        self.f_target = params[2]
        self.uleft = params[3]
        self.uright = params[4]
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
        
            dxidt = (-x_i + lambd * np.dot(W_rec, r_i) + np.dot(W_in_left, self.uleft[t]) + np.dot(W_in_right, self.uright[t]) + z_i*W_fb )/tau
            x_new = x_i + dxidt*dt
        
            P = -1.0*sp.power(z_i - self.f_target[t], 2)
            M = 1.0*(P > self.P_avg)
        
            dwdt = eta(t) * (z_i - self.z_avg) * M * r_i
            w_new = w_i + dwdt*dt

            self.P_avg = (1 - (dt/tau_avg)) * self.P_avg + (dt/tau_avg) * P
            self.z_avg = (1 - (dt/tau_avg)) * self.z_avg + (dt/tau_avg) * z_i
        
        else:
        
            dxidt = (-x_i + lambd * np.dot(W_rec, r_i) + np.dot(W_in_left, self.uleft[t]) + np.dot(W_in_right, self.uright[t]) + z_i*W_fb )/tau
                    
            x_new = x_i + dxidt*dt
            dwdt = np.zeros(np.shape(w_i))
            w_new = w_i
            
        #weight change magnitude:
        dwmag = LA.norm(dwdt)
        rmag = LA.norm(r_i)
        if t%10000 == 0: print(str(t) + '    ' + str(z_i) + '   ' + str(train))
        self.zsave[t] = z_i
        self.rsave[t] = rmag
        return np.concatenate((x_new, w_new))
            
    def run_network(self):
        for indx, t in enumerate(self.times):
            #self.xsave[indx, :] = self.state[0:Ng]
            #self.wsave[indx, :] = self.state[Ng:2*Ng]
            self.state = self.dNeurons(self.state, t)
    
# Train Parameters
dt = 1

train_trials = 90
test_trials = 10
num_trials = train_trials + test_trials

stim_length = 5000.
delay = stim_length + 10. #delay between end of one stimulus and begin of other within trial (ideal: stim_length + 10000)
iti = stim_length + 20. #time between end and begin of trials(ideal: stim_length + 20000)
len_trial = delay + iti + 2

pre_train_dur = 2000 # two seconds
train_dur = len_trial * train_trials
post_train_dur = len_trial * test_trials
dur = pre_train_dur + train_dur + post_train_dur

times = sp.arange(0.0, dur, dt)
samps = np.size(times)

training = pre_train_dur

# Generate input patterns
# Randomly generate first stim (L or R) of each trial
left_or_right = sp.zeros(num_trials) #0 indicates that the left is the first stimulus
left_or_right[sp.random.random_sample(num_trials) > 0.5] = 1 #1 indicates that the right is the first stimulus

# Make raw input streams (L and R)
delay_period = sp.zeros(delay)
iti_period = sp.zeros(iti)
same_period = sp.concatenate(([1], delay_period, [1], iti_period)) #1 is stim, 0 is no stim; stim both times
diff_period = sp.concatenate(([0], delay_period, [1], iti_period)) #stim only second time

u_left_raw = []
u_right_raw = []

for index in range(0, len(left_or_right)):
    if left_or_right[index] == 1:
        u_left_raw = sp.concatenate((u_left_raw, same_period))
        u_right_raw = sp.concatenate((u_right_raw, diff_period))
    else:
        u_left_raw = sp.concatenate((u_left_raw, diff_period))
        u_right_raw = sp.concatenate((u_right_raw, same_period))

u_left_raw = sp.concatenate((sp.zeros(pre_train_dur), u_left_raw))
u_right_raw = sp.concatenate((sp.zeros(pre_train_dur), u_right_raw))

#Target Function
f_prev = 0
def f(left, right):
    left_len = sp.size(left)
    right_len = sp.size(right)
    if left_len != right_len:
        return 0

    fret = sp.zeros(left_len)
    for samp in range(0, left_len):
        f = 0                       #0 is no stim
        if left[samp] == 1 and right[samp] == 1: #If two stims, go to opposite of prev
            f_prev = -f_prev
            f = f_prev
        elif left[samp] == 1: #0.5 is left
            f_prev = 0.5
            f = f_prev
        elif right[samp] == 1: #-0.5 is right
            f_prev = -0.5
            f = f_prev
        fret[samp] = f
    return fret
f_raw = f(u_left_raw, u_right_raw)

# Filtering
# Heaviside
def heaviside(s, intermediate):
    if intermediate ==1:
        return 1.0*(s >= 0)
    return 1.0*(s > 0)
    
# Exponential Filter
tau_l = 500. #ms
def g(s):
    return sp.exp(-s/tau_l)*heaviside(s, 1)

# H smoothing function
def h(s):
    return heaviside(s, 1) - heaviside(s - 100, 1)

filt_t = sp.linspace(-stim_length , stim_length, 2*stim_length+1)
h_filt = h(filt_t)
g_filt = g(filt_t)

# compute filtered inputs
u_left_filt = sp.convolve(heaviside(sp.convolve(u_left_raw, h_filt, 'same'), 0), g_filt, 'same')
u_right_filt = sp.convolve(heaviside(sp.convolve(u_right_raw, h_filt, 'same'), 0), g_filt, 'same')

uleft_scale_factor = 0.4/u_left_filt.max()
uright_scale_factor = 0.4/u_right_filt.max()
u_left = u_left_filt * uleft_scale_factor
u_right = u_right_filt * uright_scale_factor

f_filt = sp.convolve(f_raw, g_filt, 'same')
f_scale_factor = 0.5/f_filt.max()
f_target = f_filt*f_scale_factor

print 'Runtime: ', len(f_target)

params = (dt, training, f_target, u_left, u_right, times, svinit)

# Run network
maassnet = RNN(params)
maassnet.run_network()

# Determine performance
test_times = times[-post_train_dur:]
test_times = test_times.astype(int)
criterion = 0.25
perform = sum(1. * (abs(maassnet.zsave[test_times] - f_target[test_times]) < criterion))/len(test_times)
print 'Performance: ', str(perform * 100), '%'

plt.figure()
plt.plot(maassnet.zsave, '.')
plt.plot(maassnet.f_target, '.')
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
