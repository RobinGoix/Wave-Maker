"""
This algorithm aim at optimizing an underwater object shape to create the smaller wave as possible in SWE equations
"""

from dolfin import *
from dolfin_adjoint import * 
import pyipopt

def main(ad):  
    Ny = 3
    Nx = 200
    
    x0 = -1. #Domain [m]
    x1 = 15.
    y0 = -0.5
    y1 = 0.5
    
    g = 9.8
    hd = 1. #Depth [m]
    
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
    save = False
    ploting = True
    
    delta_t = 0.1 #timestep [s]
    t = 0.0 #time initialization
    end = 3.0 #Final Time
    delta_t = delta_t*c0/lambda0 #Time step
    t = t*c0/lambda0 #Time initialization
    end = end*c0/lambda0 #Final time

    hd = hd/h0 #depth

    #Define the profil of the moving seabed
    vmax = (hd*g)**(0.5) #Speed

    #Scaled parameters to solve the dimensionless problem
    x0 = x0/lambda0
    x1 = x1/lambda0
    y0 = y0/lambda0
    y1 = y1/lambda0
    Th = RectangleMesh(x0,y0,x1,y1,Nx,Ny)
    
    seabed = 'hd'
    traj = 'vmax*lambda0/c0*t*exp(-0.5/(lambda0/c0*t))'
    movingObject1 = 'exp(-pow((lambda0*x[0]-'+traj+')/0.3,2))'
    movingObject2 = 'exp(-pow((lambda0*x[0]+1-'+traj+')/0.3,2))'

    D = Expression(seabed,hd=hd)

    zeta1 = Expression('-' + movingObject1, hd=hd, epsilon=epsilon, lambda0=lambda0, vmax=vmax, c0=c0, t=0)
    zeta1_prev = zeta1
    zeta1_next = zeta1_prev
    
    zeta2 = Expression('-' + movingObject2, hd=hd, epsilon=epsilon, lambda0=lambda0, vmax=vmax, c0=c0, t=0)
    zeta2_prev = zeta2
    zeta2_next = zeta2_prev
    
    zeta = Expression('0.4*' + movingObject1 + ' + (1.-0.4)*' + movingObject2 , hd=hd, epsilon=epsilon, lambda0=lambda0, vmax=vmax, c0=c0, t=0)
    zeta_new = Expression('0.4*' + movingObject1 + ' + (1.-0.4)*' + movingObject2 , hd=hd, epsilon=epsilon, lambda0=lambda0, vmax=vmax, c0=c0, t=0)

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
    w_prev = Function(E, name="w^n")
    u_prev, eta_prev =  w_prev.split()

    u_prev = interpolate(u_0, V)
    eta_prev = interpolate(eta_0,H)

    w = TrialFunction(E)
    u,eta = as_vector((w[0],w[1])),w[2]

    wt = TestFunction(E)
    v,xi = as_vector((wt[0],wt[1])),wt[2]
    D = interpolate(D,H)
    zeta = interpolate(zeta,H)
    zeta_new = interpolate(zeta_new,H)
    
    zeta1_prev = interpolate(zeta1_prev,H)
    zeta1 = interpolate(zeta1,H)
    zeta1_next = interpolate(zeta1_next,H)
    
    zeta2_prev = interpolate(zeta2_prev,H)
    zeta2 = interpolate(zeta2,H)
    zeta2_next = interpolate(zeta2_next,H)

    zeta1_t = (zeta1-zeta1_prev)/delta_t
    zeta1_tt = (zeta1_next-2.*zeta1+zeta1_prev)/delta_t**2
    
    zeta2_t = (zeta2-zeta2_prev)/delta_t
    zeta2_tt = (zeta2_next-2.*zeta2+zeta2_prev)/delta_t**2

    F = 1./delta_t*inner(u-u_prev,v)*dx + epsilon*inner(grad(u)*u,v)*dx \
        - div(v)*eta*dx

    F += sigma**2.*1./delta_t*div((D+epsilon*(ad*zeta1+(1.-ad)*zeta2))*(u-u_prev))*div((D + epsilon*(ad*zeta1+(1.-ad)*zeta2))*v/2.)*dx \
      - sigma**2.*1./delta_t*div(u-u_prev)*div((D + epsilon*(ad*zeta1+(1.-ad)*zeta2))**2*v/6.)*dx \
      + sigma**2.*(ad*zeta1_tt+(1.-ad)*zeta2_tt)*div((D + epsilon*(ad*zeta1+(1.-ad)*zeta2))*v/2.)*dx

    F += 1./delta_t*(eta-eta_prev)*xi*dx + (ad*zeta1_t+(1.-ad)*zeta2_t)*xi*dx \
        - inner(u,grad(xi))*(epsilon*eta+D+epsilon*(ad*zeta1+(1.-ad)*zeta2))*dx 
        
    w_ = Function(E, name="w^{n+1}")
    (u_, eta_) = w_.split()
    F = action(F, w_)   

    ###############################ITERATIONS##########################
    while (t <= end):
        solve(F==0, w_, bc) #Solve the variational form
        w_prev.assign(w_) 
        t += float(delta_t)
        zeta1_prev.assign(zeta1)
        zeta1.assign(zeta1_next)
        zeta1_new = Expression('-' + movingObject1, \
            hd=hd, epsilon=epsilon, vmax=vmax, lambda0=lambda0, t=t, c0=c0)
        zeta1_new = interpolate(zeta1_new,H)
        zeta1_next.assign(zeta1_new)
        
        zeta2_prev.assign(zeta2)
        zeta2.assign(zeta2_next)
        zeta2_new = Expression('-' + movingObject2, \
            hd=hd, epsilon=epsilon, vmax=vmax, lambda0=lambda0, t=t, c0=c0)
        zeta2_new = interpolate(zeta2_new,H)
        zeta2_next.assign(zeta2_new)
        
        zeta.assign(zeta_new) 
        zeta_new = Expression('0.4*' + movingObject1 + ' + (1.-0.4)*' + movingObject2 , \
             hd=hd, epsilon=epsilon, lambda0=lambda0, vmax=vmax, c0=c0, t=t)
        
        if (ploting==True):
            plot(eta_,rescale=True, title = "Free Surface")
            plot(zeta,rescale=False, title = "Seabed")

        if (save==True):
            fsfile << eta_ #Save heigth
            hfile << zeta_prev
    ##############################END OF ITERATIONS#################################
    
    return w_


if __name__ == "__main__":
    ad = Constant(0.1)  #Initialisation of the value to be optimized
    #ad_new = Constant(1.)
    w_ = main(ad)   #Corresponding solution of Peregrine
    J = Functional(inner(w_[2], w_[2])*dx*dt[FINISH_TIME])  #Cost function
    #dJdad = compute_gradient(J, ScalarParameter(ad))    #Gradient of the cost function
    m = ScalarParameter(ad)
    reduced_functional = ReducedFunctional(J, m)
    m_opt = minimize(reduced_functional)
    print(float(m_opt))
    
    
    """
    w_ = main(ad)   #Corresponding solution of Peregrine
    mu = Constant(1000.) #Step to change ad
    compteur = 0
    
    while(float(mu) > 1.):
        compteur += 1
        ad = Constant(ad_new)
        w_ = main(ad)
        J = Functional(inner(w_[2], w_[2])*dx*dt[FINISH_TIME])  #Cost function
        parameters["adjoint"]["stop_annotating"] = True # stop registering equations  
        dJdad = compute_gradient(J, ScalarParameter(ad))    #Gradient of the cost function
        Jad = assemble(inner(w_[2], w_[2])*dx)  # Compute the value of the cost function
        
        ad_new = Constant(ad - mu*dJdad)
        w_new = main(ad_new)
        Jad_new = assemble(inner(w_new[2], w_new[2])*dx)
  
        if(float(Jad_new) < float(Jad)):
            mu = mu*2.
        else:
            while(float(Jad_new) > float(Jad)):
                mu = mu/2.
                ad_new = Constant(ad - mu*dJdad)
                w_new = main(ad_new)
                Jad_new = assemble(inner(w_new[2], w_new[2])*dx)
        
        print('compteur = ', compteur)       
        print('mu = ', float(mu))
        print('ad = ', float(ad))
        print('dJdad = ', float(dJdad))
        print('ad_new = ', float(ad))
"""
"""
    def Jhat(ad): # the functional as a pure function of nu
        w_ = main(ad)
        return assemble(inner(w_[2], w_[2])*dx)

    conv_rate = taylor_test(Jhat, ScalarParameter(ad), Jad, dJdad)
"""