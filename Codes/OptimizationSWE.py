"""
This algorithm aim at optimizing an underwater object shape to create the smaller wave as possible in SWE equations
"""

from dolfin import *
#from dolfin_adjoint import * 
#import pyipopt

def main(ad):  
    Ny = 2
    Nx = 200
    
    x0 = -5. #Domain [m]
    x1 = 15.
    y0 = -0.5
    y1 = 0.5
    
    g = 9.8
    hd = 1. #Depth [m]
    #ad = 0.4 #Object's height
    
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
    
    delta_t = 0.05 #timestep [s]
    t = 0.15 #time initialization
    end = 1.0 #Final Time
    delta_t = delta_t*c0/lambda0 #Time step
    t = t*c0/lambda0 #Time initialization
    end = end*c0/lambda0 #Final time

    hd = hd/h0 #depth
    #ad = ad/a0 #Object's height

    #Define the profil of the moving seabed
    vmax = (hd*g)**(0.5) #Speed

    #Scaled parameters to solve the dimensionless problem
    x0 = x0/lambda0
    x1 = x1/lambda0
    y0 = y0/lambda0
    y1 = y1/lambda0
    Th = RectangleMesh(x0,y0,x1,y1,Nx,Ny)
    plot(Th, interactive=True)
    seabed = 'hd'
    traj = 'vmax*lambda0/c0*t*exp(-0.5/(lambda0/c0*t))'
    movingObject = 'ad*0.5*(tanh(p1*(lambda0*x[0]-'+traj+'))+tanh(p2*(1-lambda0*x[0]+'+traj+')))'
    movingObject = '- (' + movingObject + ' > 0. ? ' + movingObject + ' : 0. )'

    D = Expression(seabed,hd=hd)
    #zeta = Expression(movingObject, hd=hd, epsilon=epsilon, lambda0=lambda0, vmax=vmax, c0=c0, t=0, p1=p1, p2=p2, ad=ad)
    x = triangle.x
    zeta = ad*0.5*(tanh(p1*(lambda0*x[0]-vmax*lambda0/c0*t*exp(-0.5/(lambda0/c0*t))))+tanh(p2*(1-lambda0*x[0]+vmax*lambda0/c0*t*exp(-0.5/(lambda0/c0*t)))))
    zeta_prev = zeta
    zeta_next = zeta
    
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
    
    #zeta = interpolate(zeta,H)
    #zeta_next = interpolate(zeta_next,H)
    #zeta_prev = interpolate(zeta_prev,H)
        
    zeta_t = (zeta-zeta_prev)/delta_t
    zeta_tt = (zeta_next-2.*zeta+zeta_prev)/delta_t**2

    F = 1./delta_t*inner(u-u_prev,v)*dx + epsilon*inner(grad(u)*u,v)*dx \
        - div(v)*eta*dx

    F += sigma**2.*1./delta_t*div((D+epsilon*zeta)*(u-u_prev))*div((D+epsilon*zeta)*v/2.)*dx \
            - sigma**2.*1./delta_t*div(u-u_prev)*div((D+epsilon*zeta)**2*v/6.)*dx \
            + sigma**2.*zeta_tt*div((D+epsilon*zeta)*v/2.)*dx

    F += 1./delta_t*(eta-eta_prev)*xi*dx + zeta_t*xi*dx \
        - inner(u,grad(xi))*(epsilon*eta+(D+epsilon*zeta))*dx 



    ###############################ITERATIONS##########################
    while (t <= end):
        w_ = Function(E, name="w^{n+1}")
        (u_, eta_) = w_.split()
        F = action(F, w_)   

        solve(F==0, w_, bc) #Solve the variational form
        w_prev.assign(w_) 
        t += float(delta_t)
        zeta = ad*0.5*(tanh(p1*(lambda0*x[0]-vmax*lambda0/c0*t*exp(-0.5/(lambda0/c0*t))))+tanh(p2*(1-lambda0*x[0]+vmax*lambda0/c0*t*exp(-0.5/(lambda0/c0*t)))))
        zeta_prev = ad*0.5*(tanh(p1*(lambda0*x[0]-vmax*lambda0/c0*(t-delta_t)*exp(-0.5/(lambda0/c0*(t-delta_t)))))+tanh(p2*(1-lambda0*x[0]+vmax*lambda0/c0*(t-delta_t)*exp(-0.5/(lambda0/c0*(t-delta_t))))))
        zeta_next = ad*0.5*(tanh(p1*(lambda0*x[0]-vmax*lambda0/c0*(t+delta_t)*exp(-0.5/(lambda0/c0*(t+delta_t)))))+tanh(p2*(1-lambda0*x[0]+vmax*lambda0/c0*(t+delta_t)*exp(-0.5/(lambda0/c0*(t+delta_t))))))
        zeta_t = (zeta-zeta_prev)/delta_t
        zeta_tt = (zeta_next-2.*zeta+zeta_prev)/delta_t**2

        if (ploting==True):
            plot(eta_,rescale=True, title = "Free Surface")
            #plot(zeta, Th, rescale=False, title = "Seabed")

        if (save==True):
            fsfile << eta_ #Save heigth
            hfile << zeta_prev
    ##############################END OF ITERATIONS#################################
    
    return w_


if __name__ == "__main__":
    p1 = Constant(5.)  #Initialisation of the value to be optimized
    p2 = Constant(1.)
    ad = Constant(0.1)
    #ad_new = Constant(1.)
    w_ = main(ad)   #Corresponding solution of Peregrine
    J = Functional(-1000000*inner(w_[2], w_[2])*dx*dt[FINISH_TIME])  #Cost function
    #dJdad = compute_gradient(J, ScalarParameter(ad))    #Gradient of the cost function
    m1 = ScalarParameter(ad)
    #m2 = ScalarParameter(p2)
    reduced_functional = ReducedFunctional(J, m1)
    print('reduced_functional = ', type(reduced_functional))
    #print('type([m1,m2]) = ', type([m1,m2]))
    m_opt = minimize(reduced_functional, bounds=(0.0,0.5))
    print('m_opt = ', float(m_opt))
    
    
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