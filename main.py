from lasso.dyna import D3plot, ArrayType,Binout
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os
import mat
import Force as force


#material = "T45"
material = "T65"

#Laden der Validierungssimulaitonen
MZ_folder = r"/MZ_AIMM_korrekt_t65_lcid3_lc_mit_setzen_1860el_elast"
#pfad = os.path.abspath(os.getcwd()+ "/../Simulations/MAT81_" + material + "_shl")
pfad = os.path.abspath(os.getcwd()+ "/../Simulations/" + MZ_folder)


d3plot_path = os.path.join(pfad,"d3plot")
binout_path = os.path.join(pfad,"binout")
keyword_path = os.path.join(pfad,"model.k")


mat_shl = mat.Wrapper(keyword_path,"shl")
mat_sld = mat.Wrapper(keyword_path,"sld")

first_step = 0
last_step = 40
#BINOUT
binout = Binout(binout_path)

#D3PLOT
d3plot = D3plot(d3plot_path,buffered_reading=True)
displacement_dir = "x"
exact_force = np.array(binout.read("bndout","velocity","nodes",displacement_dir+"_total"))


#Koordinaten aller Knoten im unbelasteten Zustad, shape: (Anzhal knoten, 3) x,y,z
node_coordinates = d3plot.arrays[ArrayType.node_coordinates]
#Koordinaten aller Knoten zu allen Zeitschritten, shape: (Anzhal knoten, 3) x,y,z
node_displacement = d3plot.arrays[ArrayType.node_displacement]




stresses = d3plot.arrays[ArrayType.element_shell_stress][:,:,0,:]           
#: shape (n_states, n_shells_non_rigid, n_shell_layers, xx_yy_zz_xy_yz_xz) -> stresses unterscheiden seich nicht in z dimension, da ESZ)
strains = d3plot.arrays[ArrayType.element_shell_strain][:,:,0,:]            
#: shape (n_states, n_shells_non_rigid, upper_lower, xx_yy_zz_xy_yz_xz)  -> strains unterscheiden seich nicht in z dimension, da ESZ)

t0 = np.mean(d3plot.arrays[ArrayType.element_shell_thickness][0])
#: shape (n_states, n_shells_non_rigid) 
node_coord_0 = node_displacement[0,:,:]
eps_0 = strains[0,:,:]
node_indexes = d3plot.arrays[ArrayType.element_shell_node_indexes]

#MW des Elementvolumens
volume0 = force.calculate_volume(node_indexes,node_coord_0,eps_0,t0)


"""
stresses = d3plot.arrays[ArrayType.element_solid_stress]
stresses = stresses[:,:,0,:]
#: shape: (n_states, n_solids, n_solid_layers, xx_yy_zz_xy_yz_xz) -> n_solid_layers = 1
strains = d3plot.arrays[ArrayType.element_solid_strain][:,:,0,:]            
#: shape (n_states, n_shells_non_rigid, upper_lower, xx_yy_zz_xy_yz_xz)  -> n_solid_layers = 1
"""

#Single-element-Simulation

cm = [2.45,0.38,10e03,10e03,1,1,1]

last_step = 22

strains = strains[:last_step,:,:]
stresses = stresses[:last_step,:,:] #GPa

sigmas = mat_shl.calc_stresses(strains,cm,verbose = False) #GPa
#sigmas = mat_sld.calc_stresses(strains,verbose = False)



def F_x(eps,u,force,volume, E, nu):
    """  
    Spannungsberechnung für eine elastische Deformation nach dem Hooke'schen Gesetz

    [eps_11,eps_22,eps_33, eps_12, eps_23, eps_31]
    
    Parameters
    ----------
    eps : np.array, shape: (n_states,n_ip,6)
        Dehnungen an jedem Integrationspunkt, aller Zeitschritte
        
    u : np.array(), shape: (2,1)
        Verschiebungen der ersten zwei Zeitschritte
        
    force : np.array(), shape: (2,1)
        gemessene Kraft der ersten zwei Zeitschritte
    
    volume : float
        Mittelwert der Elementvolumina
        
    E : float
        E-Modul
    nu : float
        Querkontraktionszahl

    Returns
    -------
    Func : np.array, shape: (2,)
        Funktion für das Newton-Verfahren
        
    """
      
    #bestimmen des Elastizitätstensors C
    C_11 = E*(1-nu)/((1+nu)*(1-2*nu))
    C_12 = E*nu/((1+nu)*(1-2*nu))
    C_44 = E*(1-2*nu)/((1+nu)*(1-2*nu))
    C = np.zeros([6,6])
    C[:3,:3] = C_12
    C[0,0],C[1,1],C[2,2] = C_11,C_11,C_11
    C[3,3],C[4,4],C[5,5] = C_44,C_44,C_44
    
    
    d_eps = np.gradient(eps,axis = 0)[:,:,:]
    du = np.gradient(u)
    
    Func = np.zeros(2)
    
    for j,t in enumerate([1,2]):
    
        summe = 0
        for i in range(eps.shape[1]):
            summe = summe + volume*np.dot(C@eps[t,i,:],d_eps[t,i,:])
        Func[j] =  summe - du[t]*force[t]

    return Func

def dF_dE(eps,volume,E,nu,t):
    """
    Ableitung der Funktion F (Quasistatische Energiebilanz) nach dem E-Modul

    Parameters
    ----------
    eps : np.array, shape: (n_states,n_ip,6)
        Dehnungen an jedem Integrationspunkt, aller Zeitschritte
    volume : float
        Mittelwert der Elementvolumina
    
    E : float
        E-Modul (wird durch das Newtonverfahren iterativ angepasst)
    
    nu : float
        Querkontraktionszahl (wird durch das Newtonverfahren iterativ angepasst)
        
    t : int
        timestep, entweder 1 oder 2

    Returns
    -------
    summe : float

    """
    C_11 = (1-nu)/((1+nu)*(1-2*nu))
    C_12 = nu/((1+nu)*(1-2*nu))
    C_44 = (1-2*nu)/((1+nu)*(1-2*nu))
    C = np.zeros([6,6])
    C[:3,:3] = C_12
    C[0,0],C[1,1],C[2,2] = C_11,C_11,C_11
    C[3,3],C[4,4],C[5,5] = C_44,C_44,C_44
    
    d_eps = np.gradient(eps,axis = 0)[:,:,:]
    
    summe = 0
    for i in range(eps.shape[1]):
        summe = summe + volume*np.dot(C@eps[t,i,:],d_eps[t,i,:])
    
    return summe

def dF_dnu(eps,volume,E,nu,t):
    """
    Ableitung der Funktion F (Quasistatische Energiebilanz) nach der Querkontraktionszahl

    Parameters
    ----------
    eps : np.array, shape: (n_states,n_ip,6)
        Dehnungen an jedem Integrationspunkt, aller Zeitschritte
    volume : float
        Mittelwert der Elementvolumina
    
    E : float
        E-Modul (wird durch das Newtonverfahren iterativ angepasst)
    
    nu : float
        Querkontraktionszahl (wird durch das Newtonverfahren iterativ angepasst)
        
    t : int
        timestep, entweder 1 oder 2

    Returns
    -------
    summe : float

    """
    C_11 = -2*(nu-2)*nu/(2*nu**2+nu-1)**2
    C_12 = (2*nu**2+1)/(2*nu**2+nu-1)**2
    C_44 = -1/(1+nu)**2
    C = np.zeros([6,6])
    C[:3,:3] = C_12
    C[0,0],C[1,1],C[2,2] = C_11,C_11,C_11
    C[3,3],C[4,4],C[5,5] = C_44,C_44,C_44
    
    d_eps = np.gradient(eps,axis = 0)[:,:,:]
    
    summe = 0
    for i in range(eps.shape[1]):
        summe = summe + volume*np.dot(E*C@eps[t,i,:],d_eps[t,i,:])
    
    return summe
    



u= force.displacement(node_coordinates,node_displacement,boundary = displacement_dir)
correct_force = np.array(binout.read("bndout","velocity","nodes",displacement_dir+"_total"))*1000
E_start = 1*1000 #korrekt: 2.45
rn_start = 0.45 #korrekt: 0.38
x = np.array([E_start,rn_start])


i = 0
"""
#1Dim Newton
while True:
    F = F_x(strains,u,correct_force,volume0, x[0], x[1])[0]-0.012352254229967219
    F= F_x(strains,u,correct_force,volume0, x[0], x[1])[1]-0.029571290757218982
    #F_strich = dF_dE(strains,volume0,x[0],x[1],1)
    F_strich = dF_dnu(strains,volume0,x[0],x[1],2)

    delta_x = -F/F_strich
    
    #x[0] = x[0] + delta_x
    x[1] = x[1] + delta_x
    print("Iteration: {} F: {:3.10f} E: {:3.4f} nue: {:3.10f} delta_x: {:3.10f}".format(i,F,x[0],x[1],delta_x))
    i = i + 1
    if abs(F) < 0.0000001:
        break
"""   

#2Dim Newton
while True:
    F = np.zeros(2)
    F[0] = F_x(strains,u,correct_force,volume0, x[0], x[1])[0]-0.012352254229967219 #korrektur term (abweichung aufgrund des durchschnittlichen EL. vols)
    F[1] = F_x(strains,u,correct_force,volume0, x[0], x[1])[1]-0.029571290757218982
    
    J = np.zeros([2,2])
    J[0,0] = dF_dE(strains,volume0,x[0],x[1],1)
    J[1,0] = dF_dE(strains,volume0,x[0],x[1],2)

    J[0,1] = dF_dnu(strains,volume0,x[0],x[1],1)
    J[1,1] = dF_dnu(strains,volume0,x[0],x[1],2)

    delta_x = -np.linalg.inv(J)@F
    x = x + delta_x
  
   
    print("Iteration: {} F: {:3.10f},{:3.10f}  E: {:3.4f} nue: {:3.10f} delta_x: {:3.10f}, {:3.10f}".format(i,F[0],F[1],x[0],x[1],delta_x[0],delta_x[1]))
    i = i + 1
    if abs(max(F)) < 0.0000001:
        break

    

