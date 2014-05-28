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

dt = 0.01 #timestep [s]
t = 0.0 #time initialization
end = 15.0 #Final Time
N_iter = floor(end/dt)

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
ploting = False

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
E = MixedFunctionSpace([V,H])

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

###############DEFINITION OF THE WEAK FORMULATION############

w_prev = Function(E)
(u_prev, eta_prev) = w_prev.split()

u_prev = interpolate(u_0, V)


eta_prev = interpolate(eta_0,H)
eta_c_0 = interpolate(eta_0,H)
eta_00 = interpolate(eta_0,H)
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

F = action(F, w_)   
#errorfile = File("results/SolitaryWaveDenys_dt=0.02/error.csv") #To save data in a file
error_array = []
###############################ITERATIONS##########################
while (t <= end):
    solve(F==0, w_, bc) #Solve the variational form
    (u_, eta_) = w_.split(True)
    u_prev.assign(u_) #u_prev = u_
    eta_prev.assign(eta_) #eta_prev = eta_
    t += float(dt)
    print(t)

    if (save==True):
        fsfile << eta_ #Save heigth
    
    #Computing eta_centered the translated of eta_
    N_traj = np.argmax(eta_.vector()) -np.argmax(eta_initial.vector()) #Updating the trajectory of the soliton
    eta_centered = np.zeros(4096) #Create the array of dimension 4096
    j=4095
    while(j>=0):
        if(j+N_traj <= 4095):
            eta_centered[j] = eta_.vector()[j+N_traj]
            #etae[j]=0
        j -= 1
    #Then we dedoublate the array
        
    eta_c = Function(H)
    eta_c.vector()[:] = eta_centered
    eta_moved = Expression("eta_c",eta_c=eta_c)
    eta_moved = interpolate(eta_moved,H)
    
    eta_c_0.assign(eta_moved)
    
    if (ploting==True):
        plot(eta_prev,rescale=True, title = "Free Surface")
        plot(eta_c_0, title='Solution translated')
    
    #Computing the error compare to the initial condition

    error = inner(eta_c_0 - eta_00, eta_c_0 - eta_00)*dx 
    E = sqrt(assemble(error))/sqrt(assemble(inner(eta_00,eta_00)*dx))
    error_array.append(E)
    #print(error_array)
    
    

##############################END OF ITERATIONS#################################
plt.plot(error_array, 'ro')
plt.axis([0,N_iter,0,0.2])
plt.show()