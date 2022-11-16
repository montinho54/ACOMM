from lasso.dyna import D3plot, ArrayType,Binout
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import os
import mat
import Force as force


#material = "T45"
material = "T65"

#Laden der Validierungssimulaitonen
MZ_folder = r"/MZ_AIMM_korrekt_t65_lcid3_lc_mit_setzen_1860el_elast"
MZ_folder_lager = r"/MZ_AIMM_korrekt_t65_lcid3_lc_mit_setzen_1860el_elast_opt_Lager"
rechteck_folder = r"/MAT24_T65_shell_Rechtecksprobe_optimale_Lagerung"
#pfad = os.path.abspath(os.getcwd()+ "/../Simulations/MAT81_" + material + "_shl")
#pfad = os.path.abspath(os.getcwd()+ "/../Simulations/" + MZ_folder)
#pfad = os.path.abspath(os.getcwd()+ "/../Simulations/" + rechteck_folder)
pfad = os.path.abspath(os.getcwd()+ "/../Simulations/" + MZ_folder_lager)

d3plot_path = os.path.join(pfad,"d3plot")
binout_path = os.path.join(pfad,"binout")
keyword_path = os.path.join(pfad,"model.k")



mat_shl = mat.Wrapper("shl")
mat_sld = mat.Wrapper("sld")


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
#: shape (n_states, n_shells_non_rigid, n_shell_layers, xx_yy_zz_xy_yz_xz) -> stresses unterscheiden sich nicht in z dimension, da ESZ)
strains = d3plot.arrays[ArrayType.element_shell_strain][:,:,0,:]            
#: shape (n_states, n_shells_non_rigid, upper_lower, xx_yy_zz_xy_yz_xz)  -> strains unterscheiden sich nicht in z dimension, da ESZ)

t0 = np.mean(d3plot.arrays[ArrayType.element_shell_thickness][0])
#: shape (n_states, n_shells_non_rigid) 
node_coord_0 = node_displacement[0,:,:]
eps_0 = strains[0,:,:]
node_indexes = d3plot.arrays[ArrayType.element_shell_node_indexes]


#MW des Elementvolumens
mean_el_volumes = []
probe_volumes = []

last_step = strains.shape[0]
#last_step = 50

for t in range(last_step):
    mean_el_volume, probe_volume = force.calculate_volume(node_indexes,node_displacement[t,:,:],strains[t,:,:],np.mean(d3plot.arrays[ArrayType.element_shell_thickness][t]))
    mean_el_volumes.append(mean_el_volume)
    probe_volumes.append(probe_volume)

strains = strains[:last_step,:,:]
stresses = stresses[:last_step,:,:] #GPa

cm = [2450,0.38,10e03,10e03,1,1,1]
cy = mat.read_lc_from_keyword(keyword_path, first_line_to_read = 148)
#sigmas = mat_shl.calc_stresses(strains[:,:,:],cm,cy) #GPa

def F_x(eps,u,force,volume, E, nu,t,**kwargs):
    """  
    Berechnet die Differenz zwischen der inneren Formänderungsenergie und der Arbeit
    
    Spannungsberechnung ist optional:
        für eine elastische Deformation nach dem Hooke'schen Gesetz [eps_11,eps_22,eps_33, eps_12, eps_23, eps_31]
        
    
    
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
        
    t : int
        Timestep an dem die Auswertung durchgeführt werden soll
        
    kwargs
    ------
    sig : np.array: (n_states,n_ip,6)
        Spannungen an jedem IP, aller Zeitschritte
        (Wenn vorgegeben, wird die Spannungsberechnung mittels Elastizitätstensor übersprungen)
        
        
    Returns
    -------
    Func : np.array, shape: (2,)
        Funktion für das Newton-Verfahren (Differenz zwischen der Formänderungsenergie und der Arbeit)
        
    """
    d_eps = np.gradient(eps,axis = 0)[:,:,:]
    du = np.gradient(u)
    
    Func = 0
    summe = 0
    
    if "sig" in kwargs:
        sig = kwargs["sig"]
        
        for i in range(eps.shape[1]):
            summe = summe + volume*np.dot(sig[t,i,:],d_eps[t,i,:])
        Func =  summe - du[t]*force[t]
    
    else:
        
        #bestimmen des Elastizitätstensors C
        C_11 = E*(1-nu)/((1+nu)*(1-2*nu))
        C_12 = E*nu/((1+nu)*(1-2*nu))
        C_44 = E*(1-2*nu)/((1+nu)*(1-2*nu))
        C = np.zeros([6,6])
        C[:3,:3] = C_12
        C[0,0],C[1,1],C[2,2] = C_11,C_11,C_11
        C[3,3],C[4,4],C[5,5] = C_44,C_44,C_44
        
        for i in range(eps.shape[1]):
            summe = summe + volume*np.dot(C@eps[t,i,:],d_eps[t,i,:])
        Func =  summe - du[t]*force[t]

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
    

def calc_first_principle_value(vector):
    """
    Berechnet die ersten Hauptgrößen unter der Annahme eines homogenen Spannugnszustandes in der gesamten Probe
    Größen für die diese Funktion angewendet werden kann: Tensor der Dehnunginkremente, Verzerrungstensor oder Spannungstensor
    Die Hauptgrößen entsprechen den Eigenwerten der jeweiligen Größen
    Die erste Hauptgröße wird aus den drei Eigenwerten anhand der euklidischen Norm berechnet

    Parameters
    ----------
    vector : np.array, shape: (6,)
        Beinhaltet die 6 relevanten Größen mit der Indizierung: xx,yy,zz,xy,yz,zx

    Returns
    -------
    principle_I : float
       Die ersten Hauptgrößen unter der Annahme eines homogenen Spannugnszustandes 

    """
    m = np.zeros([3,3])
    #reshapen von xx,yy,zz,xy,yz,zx in 3x3 Matrix, damit die Eigenwerte direkt berechnet werden können
    for i in range(3):
        m[i,i]=vector[i]
    m[1,0],m[0,1] = vector[3],vector[3]
    m[1,2],m[2,1] = vector[4],vector[4]
    m[2,0],m[0,2] = vector[5],vector[5]
    #berechnen der Hauptgrößen (Eigenwerte)
    principle_values = np.linalg.eig(m)[0]
    #berechnen der ersten Hauptgröße, die vorliegt, wenn ein homogenes Verzerrungsfeld vorliegen würde
    #entspricht dem Radius des Zylinders
    principle_I = np.linalg.norm(principle_values)
    principle_I = principle_values[0]
    return principle_I
    
    
u= force.displacement(node_coordinates,node_displacement,boundary = displacement_dir)
correct_force = np.array(binout.read("bndout","velocity","nodes",displacement_dir+"_total"))*1000 #F
E_start = 1*1000 #korrekt: 2.45 Gpa
nu_start = 0.45 #korrekt: 0.38
x = np.array([E_start,nu_start])
volume0 = mean_el_volumes[0]

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
    F[0] = F_x(strains,u,correct_force,volume0, x[0], x[1],t = 1)-F_x(strains,u,correct_force,volume0, 2450, 0.38,t = 1) #korrektur term ([E] = MPa, [F] = N) (Abweichung aufgrund des durchschnittlichen EL. vols)
    F[1] = F_x(strains,u,correct_force,volume0, x[0], x[1],t = 2)-F_x(strains,u,correct_force,volume0, 2450, 0.38,t = 2)
    
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

E = x[0]
nu = x[1]
d_eps = np.gradient(strains,axis = 0)[:,:,:]
du = np.gradient(u)
sig_y = []
plastic_strain = []
#d_eps : np.array, shape: (n_states,n_ip,6), zeitl. Ableitungen der Dehnungen an jedem Integrationspunkt, aller Zeitschritte

for t in range(strains.shape[0]):
    #bei einer Abweichung zwischen W_in und W_ex um > 2 J wird die approximierte Fließspannung bestimmt
    if abs(F_x(strains,u,correct_force,mean_el_volumes[t], E, nu,t)) > 2:
        
        plastic_strain.append(calc_first_principle_value(np.mean(strains[t,:,:],axis = 0)))
        d_eps_I = calc_first_principle_value(np.mean(d_eps[t,:,:],axis = 0))
        
        #direktes Auflösen der Fließspannung, nach der Energiebilanz, unter der Annahme, dass ein homogenes Spannungsfeld vorliegt
        sig_y.append(correct_force[t]*du[t]/(probe_volumes[t]*d_eps_I))
        

sig_y = np.array(sig_y)
plastic_strain = np.array(plastic_strain)
#abziehen der elastischen Dehnung (die Dehnung beim ersten mal vorliegt, wenn die Energiebilanz nicht mehr erfüllt wird)
#damit tatsächlich ausschließlich die plastische Dehnung vorliegt
plastic_strain = plastic_strain-plastic_strain[0]


cy = mat.read_lc_from_keyword(keyword_path, first_line_to_read = 148)


fig,ax = plt.subplots(figsize=(8, 8))
ax.plot(plastic_strain, sig_y, label = "calculated", linestyle = "solid", linewidth = 2, color = "k")
ax.plot(cy[:125,0],cy[:125,1], label = "correct", linestyle = "solid", linewidth = 2, color = "g")
ax.legend(loc = "lower right")
ax.grid(visible = True)
ax.set_ylim(bottom = 0)
ax.set_xlabel("$\epsilon_{p}$ [-]")
ax.set_ylabel("$\sigma_{y}$ [MPa] ")
ax.set_title("Iteration 0")

"""
###################################
"""

#beginne die fließkurveniteration
it_cy = np.concatenate((plastic_strain.reshape([-1,1]),sig_y.reshape([-1,1])),axis = 1)

sigmas_it = mat_shl.calc_stresses(strains[:,:,:],cm,it_cy) #GPa

    
dif_W = []

for t in range(sigmas_it.shape[0]):

    dif_W.append(F_x(strains,u,correct_force,mean_el_volumes[t], E, nu,t,sig = sigmas_it))


fig,ax = plt.subplots(figsize=(8, 8))
ax.plot(dif_W, label = "W_in-W_ex ", linestyle = "solid", linewidth = 2, color = "k")

ax.grid(visible = True)

ax.set_xlabel("t")
ax.set_ylabel("[J]")
ax.set_title("Energiedifferenz aus Bilanzgleichung mit Fließkurve aus Iteration 0")
