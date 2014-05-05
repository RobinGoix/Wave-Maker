"""
This code solves the Boussinesq System derived by Peregrine for a seabed of constant depth with a moving object.

"""
import matplotlib.pyplot as plt
import numpy as np
from dolfin import *

#Mesh discretization
Ny = 3
Nx = 200

#Physical values for the physical problem
g = 9.8 #Gravity [m.s^(-2)]

dt = 0.05 #timestep [s]
t = 0.0 #time initialization
end = 60.0 #Final Time

x0 = -50. #Domain [m]
x1 = 50.
y0 = -1.
y1 = 1.

hd = 1. #Depth [m]
ad = 0.5 #height of the moving object [m]
bh = 0.7 #width of the moving object 

#Scaling parameters
lambda0 = 20. #typical wavelength
a0 = 0.4 #Typical wave height
h0 = 1. #Typical depth
sigma = h0/lambda0
c0 = (h0*g)**(0.5)
epsilon = a0/h0

#Other Parameters
save = False
ploting = True

#Scaled parameters to solve the dimensionless problem
x0 = x0/lambda0
x1 = x1/lambda0
y0 = y0/lambda0
y1 = y1/lambda0
Th = RectangleMesh(x0,y0,x1,y1,Nx,Ny,'crossed')

dt = dt*c0/lambda0 #Time step
t = t*c0/lambda0 #Time initialization
end = end*c0/lambda0 #Final time

hd = hd/h0 #depth
ad = ad/a0 #height of the moving object

#Define the profil of the moving seabed
vmax = (hd*h0*g)**(0.5)/2 #Speed
U = h0/(c0*a0)*vmax #Rescaled speed
vObject = Expression(("U","0.0"), U=U, c0=c0)
"""
seabed = 'hd - 0.5*lambda0/10.*(x[1]>2./lambda0 ? 1. : 0.)*(x[1]-2./lambda0) + 0.5*lambda0/10.*(x[1]<(-2./lambda0) ? 1. : 0.)*(x[1]+2./lambda0)'
movigObject = ' - (x[1]>0 ? 1. : 0.)*epsilon*ad*0.5*0.5*(1. - tanh(lambda0*x[1]-2.))*(tanh(10*(1. - lambda0*x[0])) + tanh(lambda0*x[0] + 1)) ' \
            + ' - (x[1]<=0 ? 1. : 0.)*epsilon*ad*0.5*0.5*(1. + tanh(lambda0*x[1]+2.))*(tanh(10*(1. - lambda0*x[0])) + tanh(lambda0*x[0] + 1)) ' 
"""

seabed = 'hd'
movingObject = '- 0.25*1/pow(cosh(pow(3*0.5,0.5)*lambda0*x[0]/2),2)'

#movingObject = '- epsilon*ad*0.5*(tanh(10*(1. - lambda0*x[0])) + tanh(lambda0*x[0] + 1)) '
bottom = seabed + movingObject 
h = Expression(bottom, hd=hd, ad=ad, epsilon=epsilon, bh=bh, lambda0=lambda0, vmax=vmax, c0=c0, t=0)

#Saving parameters
if (save==True):
    fsfile = File("results/PeregrineSolitaryWave5/FS.pvd") #To save data in a file
    hfile = File("results/PeregrineSolitaryWave5/MB.pvd") #To save data in a file

#Define functions spaces
#Velocity
V = VectorFunctionSpace(Th,"Lagrange",2)
#Height
H = FunctionSpace(Th, "Lagrange", 1) 
E = V * H

#Dirichlet BC

def NoSlip_boundary(x, on_boundary):
        return on_boundary and \
               (x[1] < y0 + DOLFIN_EPS or x[1] > y1 - DOLFIN_EPS or \
                x[0] < x0 + DOLFIN_EPS or x[0] > x1 - DOLFIN_EPS)
No_Slip0 = DirichletBC(E.sub(0),  Expression(("-U","0.0"), U=U), NoSlip_boundary)

def Entry_boundary(x, on_boundary):
        return on_boundary and x[0] > x1 - DOLFIN_EPS
No_Slip1 = DirichletBC(E.sub(1),  Expression(("0.0"), epsilon=epsilon, c0=c0), Entry_boundary)
bc = [No_Slip0,No_Slip1]

n=FacetNormal(Th) #Normal Vector

###############DEFINITION OF THE WEAK FORMULATION############
u_0 = Expression(("-U", "0.0"), U=U) #Initialisation of the velocity
#eta_0 = Expression("ad*1/pow(cosh(pow(3*0.5,0.5)*lambda0*x[0]/2),2)",epsilon=epsilon, ad=ad,lambda0=lambda0) #Initialisation of the free surface

eta_0 = Expression("0.0")
filtre_eta = Expression("0.5*(x[0]<=(x0+20/lambda0)? 1. : 0.)*(1+cos(pi*(lambda0*x[0]-(lambda0*x0+20))/20)) + (x[0]>(x0+20/lambda0)? 1. : 0.)",x0=x0,lambda0=lambda0)
filtre_eta = interpolate(filtre_eta,H)
#filtre_u = Expression(("0.5*(x<(x0+20)/lambda0 ? 1. : 0.)*(1+cos(pi*(x-(x0+20))/20)) + (x>(x0+20)? 1. : 0.)", 1.0),x0=x0,lambda0=lambda0)
w_prev = Function(E)
(u_prev, eta_prev) = w_prev.split()

u_prev = interpolate(u_0, V)

eta_prev = interpolate(eta_0,H)

vObject = interpolate(vObject,V)

h = interpolate(h,H)

w = TrialFunction(E)
u,eta = as_vector((w[0],w[1])),w[2]

wt = TestFunction(E)
v,xi = as_vector((wt[0],wt[1])),wt[2]

F = 1./dt*inner(u-u_prev,v)*dx + inner(grad(u)*(epsilon*(u-vObject)),v)*dx \
    + inner(v,grad(eta))*dx

#F += sigma**2.*1./dt*div(h*(u-u_prev))*div(h*v/2.)*dx \
#      - sigma**2.*1./dt*div(u-u_prev)*div(h*h*v/6.)*dx

F += 1./dt*(eta-eta_prev)*xi*dx + div((h+epsilon*eta)*(u-vObject))*xi*dx

w_ = Function(E)
(u_, eta_) = w_.split()
F = action(F, w_)   

###############################ITERATIONS##########################
while (t <= end):
    solve(F==0, w_, bc) #Solve the variational form
    eta = project(eta_*filtre_eta,H)
    u_prev.assign(u_) #u_prev = u_
    eta_prev.assign(eta) #eta_prev = eta_
    t += float(dt)
    print(t)
    if (ploting==True):
        #plot(filtre_eta)
        plot(eta_prev,rescale=True, title = "Free Surface")
        #plot(h,rescale=False, title = "Seabed")
        #plot(u_,rescale = True)
    if (save==True):
        fsfile << eta_ #Save heigth
        hfile << h

##############################END OF ITERATIONS#################################
