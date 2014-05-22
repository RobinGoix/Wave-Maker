"""
This code solves the Boussinesq System derived by Peregrine for a seabed of constant depth with a moving object.

"""
import scipy.integrate as si
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from dolfin import *

#Mesh discretization
Ny = 1
Nx = 2047

#Physical values for the physical problem
g = 1. #Gravity [m.s^(-2)]

dt = 0.02 #timestep [s]
t = 0.0 #time initialization
end = 30.0 #Final Time

x0 = -40. #Domain [m]
x1 = 40.
y0 = -0.5
y1 = 0.5

hd = 1. #Depth [m]
ad = 1. #height of the moving object [m]
bh = 0.7 #width of the moving object 
xh = 0.0 #start position of the moving object [m]

#Scaling parameters
lambda0 = 1. #typical wavelength
a0 = 1. #Typical wave height
h0 = 1. #Typical depth
sigma = h0/lambda0
c0 = (h0*g)**(0.5)
epsilon = a0/h0

#Other Parameters
save = False
moving = False
ploting = True

#Scaled parameters to solve the dimensionless problem
x0 = x0/lambda0
x1 = x1/lambda0
y0 = y0/lambda0
y1 = y1/lambda0
Th = RectangleMesh(x0,y0,x1,y1,Nx,Ny)

dt = dt*c0/lambda0 #Time step
t = t*c0/lambda0 #Time initialization
end = end*c0/lambda0 #Final time

hd = hd/h0 #depth
ad = ad/a0 #height of the moving object
xh = xh/lambda0 #start position of the moving object

#Define the profil of the moving seabed
bottom = 'hd'
h = Expression(bottom, hd=hd) 
  
#Saving parameters
if (save==True):
    fsfile = File("results/SolitaryWaveDenys_dt=0.02/FS.pvd") #To save data in a file
    

#Define functions spaces
#Velocity
V = VectorFunctionSpace(Th,"Lagrange",1)
#Height
H = FunctionSpace(Th, "Lagrange", 1) 
E = V * H

#Dirichlet BC

def NoSlip_boundary(x, on_boundary):
        return on_boundary and \
               (x[1] < y0 + DOLFIN_EPS or x[1] > y1 - DOLFIN_EPS)# or \
               # x[0] < x0 + DOLFIN_EPS or x[0] > x1 - DOLFIN_EPS)
No_Slip = DirichletBC(E.sub(0).sub(1), 0.0, NoSlip_boundary)

bc = No_Slip
n=FacetNormal(Th) #Normal Vector

#Initial Conditions
eta0 = genfromtxt('eta.txt')[np.newaxis]
eta0 = eta0[0]
eta00 = np.zeros(4096)

u0 = genfromtxt('u.txt')[np.newaxis]
u0 = u0[0]
u00 = np.zeros(4096)

i = 0
while(i<=4094):
    i += 1
    eta00[i] = eta0[floor(i/2.)]
    u00[i] = u0[floor(i/2.)]

eta_initial = Function(H)
eta_initial.vector()[:] = eta00
eta_0 = Expression("eta_initial",eta_initial=eta_initial)

u_initial = Function(H)
u_initial.vector()[:]=u00
u_0=Function(V)
u_0=Expression(("u_initial","0.0"),u_initial=u_initial)
"""
plot(eta_0,Th)
plot(u_0,Th)
interactive()
"""
###############DEFINITION OF THE WEAK FORMULATION############

w_prev = Function(E)
(u_prev, eta_prev) = w_prev.split()

u_prev = interpolate(u_0, V)

eta_prev = interpolate(eta_0,H)
eta_moved = interpolate(eta_0,H)
h = interpolate(h,H)

w = TrialFunction(E)
u,eta = as_vector((w[0],w[1])),w[2]

wt = TestFunction(E)
v,xi = as_vector((wt[0],wt[1])),wt[2]

F = 1./dt*inner(u-u_prev,v)*dx + epsilon*inner(grad(u)*u,v)*dx \
    - div(v)*eta*dx

F += sigma**2.*1./dt*div(h*(u-u_prev))*div(h*v/2.)*dx \
      - sigma**2.*1./dt*div(u-u_prev)*div(h*h*v/6.)*dx

F += 1./dt*(eta-eta_prev)*xi*dx - inner(u,grad(xi))*(epsilon*eta+h)*dx 
     
    
w_ = Function(E)
(u_, eta_) = w_.split()
F = action(F, w_)   
traj = 0.
delta_x = 80./2048.
###############################ITERATIONS##########################
while (t <= end):
    solve(F==0, w_, bc) #Solve the variational form
    u_prev.assign(u_) #u_prev = u_
    eta_prev.assign(eta_) #eta_prev = eta_
    t += float(dt)
    print(t)

    if (ploting==True):
        plot(eta_,rescale=True, title = "Free Surface")

    if (save==True):
        fsfile << eta_ #Save heigth
    
    #Computing eta_e the translated eta_0
    traj += w_.vector().max()*delta_x #Updating the trajectory of the soliton
    etae = eta0 #Create the array of dimension 2048
    N_traj = int(floor(traj/delta_x))
    j=2047
    while(j>=0):
        if(j+N_traj <= 2047):
            etae[j+N_traj] = eta0[j]
            #etae[j]=0
        j -= 1
    #Then we dedoublate the array
    etaee = np.zeros(4096)
    #print(eta)
    k = 0
    while(k<=4094):
        k += 1
        etaee[k] = etae[floor(k/2.)]
        
    eta_e = Function(H)
    eta_e.vector()[:] = etaee
    eta_0_moved = Expression("eta_e",eta_e=eta_e)
    eta_0_moved = interpolate(eta_0_moved,H)
    
    eta_moved.assign(eta_0_moved)
    plot(eta_moved, title='Real Solution')
    
    #Computing the error compare to the initial condition
    errorfile = File("results/SolitaryWaveDenys_dt=0.02/error.pvd") #To save data in a file
    error = inner(eta_ - eta_0_moved, eta_ - eta_0_moved)*dx 
   # E = sqrt(assemble(error))/sqrt(assemble(inner(eta_0,eta_0)*dx))
    #Other solution : difference des maximas
    #errorfile << E
    
    

##############################END OF ITERATIONS#################################
