import numpy as np

kappa = 1.0
phi = 0.25
alpha = 1.0
A = 1.0
T = 10.0
E0 = 10.0

# constants of the closed-form solution of ODE (17)
theta = np.sqrt(kappa*phi + alpha**2/16) / kappa
r_plus = -alpha/(4*kappa) + theta
r_minus = -alpha/(4*kappa) - theta
a = (alpha/4 + kappa*theta - A) * np.exp(-theta*T) / (
    -alpha/2*np.sinh(theta*T) + 2*kappa*theta*np.cosh(theta*T) + 2*A*np.sinh(theta*T))

def E(t):
    # mean inventory of the crowd
    return E0*a*(np.exp(r_plus*t) - np.exp(r_minus*t)) + E0*np.exp(r_minus*t)

def dE(t):
    # its derivative = crowd's flow mu(t)
    return E0*a*(r_plus*np.exp(r_plus*t) - r_minus*np.exp(r_minus*t)) + E0*r_minus*np.exp(r_minus*t)


if __name__ == "__main__":
    print(E(5))                      
    print(E(T))                      
    print(-dE(0))
    # must be zero for T              
    print(2*kappa*dE(T) + 2*A*E(T)) # (Remark 3.2) at t=T: h1(T) = 2*kappa*E'(T) + h2(T)*E(T), with h1(T)=0, h2(T)=2A