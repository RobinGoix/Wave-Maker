"""
This code solves the Boussinesq System derived by Peregrine for a seabed of constant depth with a moving object.

"""
import scipy.integrate as si
import matplotlib.pyplot as plt
import numpy as np
from dolfin import *

#Mesh discretization
Ny = 60
Nx = 100

#Physical values for the physical problem
g = 9.8 #Gravity [m.s^(-2)]

dt = 0.05 #timestep [s]
t = 0.0 #time initialization
end = 60.0 #Final Time

x0 = -4. #Domain [m]
x1 = 40.
y0 = -15.
y1 = 15.

hd = 1. #Depth [m]
ad = 0.4 #height of the moving object [m]
bh = 0.7 #width of the moving object 
xh = 0.0 #start position of the moving object [m]

u_0 = Expression(("0.0", "0.0")) #Initialisation of the velocity
eta_0 = Expression("0.0") #Initialisation of the free surface

#Scaling parameters
lambda0 = 20. #typical wavelength
a0 = 0.4 #Typical wave height
h0 = 1. #Typical depth
sigma = h0/lambda0
c0 = (h0*g)**(0.5)
epsilon = a0/h0

#Other Parameters
save = True
moving = True
ploting = True

#Scaled parameters to solve the dimensionless problem
x0 = x0/lambda0
x1 = x1/lambda0
y0 = y0/lambda0
y1 = y1/lambda0
Th = RectangleMesh(x0,y0,x1,y1,Nx,Ny)

#Refine Mesh along the object's trajectory
cell_markers = CellFunction("bool", Th)
cell_markers.set_all(False)
for cell in cells(Th):
    p = cell.midpoint()
    if p.y() > -3.5/20 and p.y() < 3.5/20 :
        cell_markers[cell] = True   
Th = refine(Th, cell_markers)

cell_markers2 = CellFunction("bool", Th)
cell_markers2.set_all(False)
for cell in cells(Th):
    p = cell.midpoint()
    if p.y() > -3.5/20 and p.y() < 3.5/20 :
        cell_markers2[cell] = True   
Th = refine(Th, cell_markers2)

dt = dt*c0/lambda0 #Time step
t = t*c0/lambda0 #Time initialization
end = end*c0/lambda0 #Final time

hd = hd/h0 #depth
ad = ad/a0 #height of the moving object
xh = xh/lambda0 #start position of the moving object

#Define the profil of the moving seabed
if (moving == True):
    #Parameters for the velocity
    """
    p1=4
    d1=1.
    p2=3.
    d2=4.
    vfinal = 1.5 #Maximal velocity of the moving object [m.s^(-1)]
    velocity = lambda tt: 0.5*vfinal*(tanh(p1*(lambda0/c0*tt-d1))+tanh(p2*(d2-(lambda0/c0)*tt)))
    vh = velocity(dt)

    #Plotting velocity curve
    r=np.arange(t,end,dt)
    plt.plot(r,map(velocity, r))
    plt.show()
    """

    xfinal = 30. #Final Position of the moving object [m]
    vmax = (hd*g)**(0.5) #Speed
    #traj = '(c0*vfinal*(log(tanh((3*lambda0*t)/c0 - 6) + 1) - log(tanh((3*lambda0*t)/c0 - 12) + 1) - log(tanh((3*lambda0*t0)/c0 - 6) + 1) + log(tanh((3*lambda0*t0)/c0 - 12) + 1)))/(6*lambda0)'
    #traj = 'xfinal/2.*(tanh((lambda0/c0*t-2.)*2*vmax/xfinal)+1.-tanh(4.*2.*3./30.))'
    seabed = 'hd - 0.5/10.*(x[1]>4./lambda0 ? 1. : 0.)*(lambda0*x[1]-4.) + 0.5/10.*(x[1]<(-4./lambda0) ? 1. : 0.)*(lambda0*x[1]+4.)'
    traj = 'vmax*lambda0/c0*t*exp(-0.001/pow(lambda0/c0*t,2))'
    movingObject = ' - (x[1]<3/lambda0 ? 1. : 0.)*(x[1]>0 ? 1. : 0.)*(lambda0*x[0]-'+traj+'>-6 ? 1. : 0.)*epsilon*ad*0.5*0.5*(1. - tanh(0.5*lambda0*x[1]-2.))*(tanh(10*(1. - (lambda0*x[0] - ' + traj + ')-pow(lambda0*x[1],2)/5)) + tanh(2*((lambda0*x[0] - ' + traj + ')+pow(lambda0*x[1],2)/5 + 0.5))) ' \
                  + ' - (x[1]>-3/lambda0 ? 1. : 0.)*(x[1]<=0 ? 1. : 0.)*(lambda0*x[0]-'+traj+'>-6 ? 1. : 0.)*epsilon*ad*0.5*0.5*(1. + tanh(0.5*lambda0*x[1]+2.))*(tanh(10*(1. - (lambda0*x[0] - ' + traj + ')-pow(lambda0*x[1],2)/5)) + tanh(2*((lambda0*x[0] - ' + traj + ')+pow(lambda0*x[1],2)/5 + 0.5))) ' 
    #seabed = 'hd'
    #movingObject = '- 0.25*1/pow(cosh(pow(3*0.5,0.5)*(lambda0*x[0] - lambda0/c0*vmax*t)/2),2)'
    bottom = seabed + movingObject
    
    h_prev = Expression(bottom, hd=hd, ad=ad, epsilon=epsilon, xh=xh, bh=bh, lambda0=lambda0, vmax=vmax, xfinal=xfinal, c0=c0, t=0)
    h = h_prev
    h_next = h_prev
    
else:
    h_prev = Constant(hd)
    h = Constant(hd)
    h_next = Constant(hd)
  
#Saving parameters
if (save==True):
    fsfile = File("results/Objectshape1/FS.pvd") #To save data in a file
    hfile = File("results/Objectshape1/MB.pvd") #To save data in a file

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
No_Slip = DirichletBC(E.sub(0), [0.0, 0.0], NoSlip_boundary)

bc = No_Slip

n=FacetNormal(Th) #Normal Vector

###############DEFINITION OF THE WEAK FORMULATION############

w_prev = Function(E)
(u_prev, eta_prev) = w_prev.split()

u_prev = interpolate(u_0, V)

eta_prev = interpolate(eta_0,H)

h_prev = interpolate(h_prev,H)
h = interpolate(h,H)
h_next = interpolate(h_next,H)

w = TrialFunction(E)
u,eta = as_vector((w[0],w[1])),w[2]

wt = TestFunction(E)
v,xi = as_vector((wt[0],wt[1])),wt[2]

zeta_t = (h-h_prev)/dt
zeta_tt = (h_next-2.*h+h_prev)/dt**2

F = 1./dt*inner(u-u_prev,v)*dx + epsilon*inner(grad(u)*u,v)*dx \
    - div(v)*eta*dx

F += sigma**2.*1./dt*div(h*(u-u_prev))*div(h*v/2.)*dx \
      - sigma**2.*1./dt*div(u-u_prev)*div(h*h*v/6.)*dx \
      + sigma**2.*1./epsilon*zeta_tt*div(h*v/2.)*dx

F += 1./dt*(eta-eta_prev)*xi*dx + 1./epsilon*zeta_t*xi*dx \
      - inner(u,grad(xi))*(epsilon*eta+h)*dx 
     
    
w_ = Function(E)
(u_, eta_) = w_.split()
F = action(F, w_)	

###############################ITERATIONS##########################
while (t <= end):
    #solve(F==0, w_, bc) #Solve the variational form
    #u_prev.assign(u_) #u_prev = u_
    #eta_prev.assign(eta_) #eta_prev = eta_
    t += float(dt)
    print(t)

    if(moving==True): #Move the object --> assigne new values to h_prev, h_, h_next
        h_prev.assign(h)
        h.assign(h_next)
        #intvh=si.quad(velocity, 0, t)
        #intvh=intvh[0] 
        h_new = Expression(bottom, \
            hd=hd, ad=ad, epsilon=epsilon, xh=xh, bh=bh, xfinal=xfinal, vmax=vmax, lambda0=lambda0, t=t, c0=c0)
        h_new = interpolate(h_new,H)
        h_next.assign(h_new)

    if (ploting==True):
        #plot(eta_,rescale=True, title = "Free Surface")
        plot(h_next,rescale=False, title = "Seabed")

    if (save==True):
        fsfile << eta_ #Save heigth
        hfile << h_prev

##############################END OF ITERATIONS#################################
