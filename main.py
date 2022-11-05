from lasso.dyna import D3plot, ArrayType,Binout
import numpy as np
import matplotlib.pyplot as plt

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

"""
Kraftberechnung
"""

"""
correct_F,calc_F = [],[]
u= force.displacement(node_coordinates,node_displacement,boundary = displacement_dir)
for step in range(22):

    du = np.gradient(u)[step]
    
    correct_F.append(np.array(binout.read("bndout","velocity","nodes",displacement_dir+"_total"))[step])
    
    d_eps = np.gradient(strains,axis = 0)[step,:,:]
    sigmas_force = np.copy(sigmas[step,:,:]) #GPa
    dW_in = force.calc_dW_in(sigmas_force,d_eps,volume0)
    calc_F.append(dW_in/du) #kN
    cF = np.array(calc_F[1:])
    coF = np.array(correct_F[1:])
    ratio = cF/coF
"""




def hook_numpy(d, em, rn):
    """  
    Spannungsberechnung für eine elastische Deformation nach dem Hooke'schen Gesetz
    Materialmodell (hook_numpy) erwartet Gleitungen:
    [eps_11,eps_22,eps_33, 2*eps_12, 2*eps_23, 2*eps_31]
    
    Parameters
    ----------
    d : np.array, shape: (n_ip,6)
        Dehnungen an jedem Integrationspunkt
    em : float
        E-Modul
    rn : float
        Querkontraktionszahl

    Returns
    -------
    s : np.array, shape: (n_ip,6)
        

    """
    s = np.zeros(d.shape)
    
    d_gleitungen = np.copy(d)
    d_gleitungen[:,3:] = 2*np.copy(d_gleitungen[:,3:])

    gm = em / (2.0*(1.0+rn))
    bk = em / (3.0*(1.0-2.0*rn))
    ev = (np.sum(d_gleitungen[:,:3], axis=1)/3.0).reshape([d_gleitungen.shape[0],1])
    evv = np.concatenate([ev,ev,ev],1)

    s[:,:3] = 3.0*bk*evv + 2.0 *gm*(d_gleitungen[:,:3]-evv)
    s[:,3:] = gm *d_gleitungen[:,3:]
    return s

def F_x(eps,u,force,volume, E, rn):
    """  
    Spannungsberechnung für eine elastische Deformation nach dem Hooke'schen Gesetz
    Materialmodell (hook_numpy) erwartet Gleitungen:
    [eps_11,eps_22,eps_33, 2*eps_12, 2*eps_23, 2*eps_31]
    
    Parameters
    ----------
    eps : np.array, shape: (2,n_ip,6)
        Dehnungen an jedem Integrationspunkt, der ersten zwei Zeitschritte
        
    u : np.array(), shape: (2,1)
        Verschiebungen der ersten zwei Zeitschritte
        
    force : np.array(), shape: (2,1)
        gemessene Kraft der ersten zwei Zeitschritte
    
    volume : float
        MW des Elementvolumens
        
    E : float
        E-Modul
    rn : float
        Querkontraktionszahl

    Returns
    -------
    s : np.array, shape: (n_ip,6)
        
    """
      
    #d_gleitungen = np.copy(eps)
    eps[:,:,3:] = 2*np.copy(eps[:,:,3:])
    eps_1 = eps[1,:,:]
    eps_2 = eps[2,:,:]
    
    sig_1 = np.zeros(eps_1.shape)
    sig_2 = np.zeros(eps_2.shape)
    
    #Spannungsberechnung
    gm = E / (2.0*(1.0+rn))
    bk = E / (3.0*(1.0-2.0*rn))
    
    #berechnen von sig_1
    ev = (np.sum(eps_1[:,:3], axis=1)/3.0).reshape([eps_1.shape[0],1])
    evv = np.concatenate([ev,ev,ev],1)

    sig_1[:,:3] = 3.0*bk*evv + 2.0 *gm*(eps_1[:,:3]-evv)
    sig_1[:,3:] = gm *eps_1[:,3:]
    
    #berechnen von sig_2
    ev = (np.sum(eps_2[:,:3], axis=1)/3.0).reshape([eps_2.shape[0],1])
    evv = np.concatenate([ev,ev,ev],1)

    sig_2[:,:3] = 3.0*bk*evv + 2.0 *gm*(eps_2[:,:3]-evv)
    sig_2[:,3:] = gm *eps_2[:,3:]
 
    
    d_eps = np.gradient(strains,axis = 0)[:,:,:]
    d_eps_1 = d_eps[1,:,:]
    d_eps_2 = d_eps[2,:,:]
    
    du = np.gradient(u)
    du_1 = du[1]
    du_2 = du[2]
    
    p = np.zeros(eps_1.shape[0])
    for i in range(eps_1.shape[0]):
        p[i] = np.dot(sig_1[i,:],d_eps_1[i,:])
    F_1 = volume*np.sum(p)-force[1]*du_1
    
    p = np.zeros(eps_2.shape[0])
    for i in range(eps_2.shape[0]):
        p[i] = np.dot(sig_2[i,:],d_eps_2[i,:])
    F_2 = volume*np.sum(p)-force[2]*du_2

    F = np.array([F_1,F_2])
    return F

def jacobian(eps ,u,volume, E, rn):
    """  
    Berechnet die Jacobimatrix für das mehrdimensionale Newtonverfahren zur Ermittlung von em und rn
    Spannungsberechnung für eine elastische Deformation nach dem Hooke'schen Gesetz
    Materialmodell (hook_numpy) erwartet Gleitungen:
    [eps_11,eps_22,eps_33, 2*eps_12, 2*eps_23, 2*eps_31]
    
    Parameters
    ----------
    eps : np.array(), shape: (2,n_ip,6)
        Dehnungen der ersten zwei Zeitschritte (xx,yy,zz,xy,yz,zx)
        
      
    u :np.array(), shape: (2,1)
        Verschiebungen der ersten zwei Zeitschritte
    
    volume : float
        MW des Elementvolumens
    
    E : float
        E-Modul
    rn : float
        Querkontraktionszahl

    Returns
    -------
    J : np.array, shape: (2,2)
        Jacobimatrix
        
    """
    
    #d_gleitungen = np.copy(eps)
    eps[:,:,3:] = 2*np.copy(eps[:,:,3:])
    eps_1 = eps[1,:,:]
    eps_2 = eps[2,:,:]
    
    d_eps = np.gradient(strains,axis = 0)[:,:,:]
    d_eps_1 = d_eps[1,:,:]
    d_eps_2 = d_eps[2,:,:]
    
    du = np.gradient(u)
    du_1 = du[1]
    du_2 = du[2]
    
    #Ableitungen des Schubmoduls gm und des Kompressionsmoduls bk nach den gesuchten Größen (E und rn)

    dgm_dE = 1 / (2.0*(1.0+rn))
    dbk_dE = 1 / (3.0*(1.0-2.0*rn))
    
    dgm_drn = E / (2.0*(1.0+rn)**2)
    dbk_drn = 2*E / (3.0*(1.0-2.0*rn)**2)
        
    ev_1 = (np.sum(eps_1[:,:3], axis=1)/3.0).reshape([eps_1.shape[0],1])
    evv_1 = np.concatenate([ev_1,ev_1,ev_1],1)
    
    ev_2 = (np.sum(eps_2[:,:3], axis=1)/3.0).reshape([eps_2.shape[0],1])
    evv_2 = np.concatenate([ev_2,ev_2,ev_2],1)
    
    dsig1_dE, dsig2_dE, dsig1_drn, dsig2_drn = np.zeros(d_eps_1.shape),np.zeros(d_eps_1.shape),np.zeros(d_eps_1.shape),np.zeros(d_eps_1.shape)


    dsig1_dE[:,:3] = 3.0*dbk_dE*evv_1 + 2.0 *dgm_dE*(eps_1[:,:3]-evv_1)
    dsig1_dE[:,3:] = dgm_dE *eps_1[:,3:]
    dF1_dE = np.zeros(len(dsig1_dE))
    for i in range(len(dF1_dE)):
        dF1_dE[i] = np.dot(dsig1_dE[i,:],d_eps_1[i,:])
    dF1_dE = volume*np.sum(dF1_dE)
    
    dsig2_dE[:,:3] = 3.0*dbk_dE*evv_2 + 2.0 *dgm_dE*(eps_2[:,:3]-evv_2)
    dsig2_dE[:,3:] = dgm_dE *eps_2[:,3:]
    dF2_dE = np.zeros(len(dsig1_dE))
    for i in range(len(dF2_dE)):
        dF2_dE[i] = np.dot(dsig2_dE[i,:],d_eps_2[i,:])
    dF2_dE = volume*np.sum(dF2_dE)
    
    dsig1_drn[:,:3] = 3.0*dbk_drn*evv_1 + 2.0 *dgm_drn*(eps_1[:,:3]-evv_1)
    dsig1_drn[:,3:] = dgm_drn *eps_1[:,3:]
    dF1_drn = np.zeros(len(dsig1_dE))
    for i in range(len(dF1_drn)):
        dF1_drn[i] = np.dot(dsig1_drn[i,:],d_eps_1[i,:])
    dF1_drn = volume*np.sum(dF1_drn)
    
    dsig2_drn[:,:3] = 3.0*dbk_drn*evv_2 + 2.0 *dgm_drn*(eps_2[:,:3]-evv_2)
    dsig2_drn[:,3:] = dgm_drn *eps_2[:,3:]
    dF2_drn = np.zeros(len(dsig1_dE))
    for i in range(len(dF2_drn)):
        dF2_drn[i] = np.dot(dsig2_drn[i,:],d_eps_2[i,:])
    dF2_drn = volume*np.sum(dF2_drn)
    
    
    J = np.array([[dF1_dE,dF1_drn],[dF2_dE,dF2_drn]])
    
    return J


u= force.displacement(node_coordinates,node_displacement,boundary = displacement_dir)
correct_F = np.array(binout.read("bndout","velocity","nodes",displacement_dir+"_total"))
E_start = 5
rn_start = 0.2
x = np.array([E_start,rn_start])


F = F_x(strains,u,correct_F,volume0, x[0], x[1])
J = jacobian(strains ,u,volume0, x[0], x[1])

delta_x = -np.linalg.inv(J)@F
x = x + delta_x
norm = []
norm.append(np.linalg.norm(delta_x))
i = 0

while np.linalg.norm(delta_x) > 5:
    F = F_x(strains,u,correct_F,volume0, x[0], x[1])
    J = jacobian(strains ,u,volume0, x[0], x[1])

    delta_x = -np.linalg.inv(J)@F
    
    norm.append(np.linalg.norm(delta_x))
    
    x = x + delta_x
    print("Iteration: {} E: {:3.4f} nue: {:3.4f} norm: {:3.4f}".format(i,x[0],x[1],np.linalg.norm(delta_x)))
    i = i + 1


