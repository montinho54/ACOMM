from lasso.dyna import D3plot, ArrayType,Binout
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
import os
import mat
import Force as force


#Laden der Validierungssimulaitonen

MZ_T65_gesperrt = r"MZ_t65_lc_mit_setzen_1800el_gesperrt"
MZ_T65_offen = r"MZ_t65_lc_mit_setzen_1800el_offen"

rechteck_T65_gesperrt = r"rechteck_t65_lc_mit_setzen_1800el_gesperrt"
rechteck_T65_offen = r"rechteck_t65_lc_mit_setzen_1800el_offen"

pfad = os.path.abspath(os.getcwd()+ "/../Simulations/" + MZ_T65_gesperrt)
titel = "taillierte Zugprobe (T65) nicht zwängungsfrei gelagert"




d3plot_path = os.path.join(pfad,"d3plot")
binout_path = os.path.join(pfad,"binout")
keyword_path = os.path.join(pfad,"model.k")

#rechteck 85
#taillierte probe 120
cy = mat.read_lc_from_keyword(keyword_path, first_line_to_read = 85)

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

for t in range(last_step):
    mean_el_volume, probe_volume = force.calculate_volume(node_indexes,node_displacement[t,:,:],strains[t,:,:],np.mean(d3plot.arrays[ArrayType.element_shell_thickness][t]))
    mean_el_volumes.append(mean_el_volume)
    probe_volumes.append(probe_volume)

strains = strains[:last_step,:,:]
stresses = stresses[:last_step,:,:] #GPa


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
    

def calc_effective_strain(vector):
    """
    Berechnet die Vergleichsdehnung

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
    
    principle_I = np.sqrt((2/3)*np.einsum("ij,ij",m,m))
    
    return principle_I
    
    
u= force.displacement(node_coordinates,node_displacement,boundary = displacement_dir)
correct_force = np.array(binout.read("bndout","velocity","nodes",displacement_dir+"_total"))*1000 #F
E_start = 500 #korrekt: 2.45 Gpa
nu_start = 0.45 #korrekt: 0.38
x = np.array([E_start,nu_start])
volume0 = mean_el_volumes[0]

E_iter = []
nu_iter = []

i = 0
 

#2Dim Newton
while True:
    F = np.zeros(2)
    #T65 E: 2.45 nu: 0.38, sigy:   0.0377, rho: 1.13000E-6
    #T45 E: 2.1 nu: 0.38, sigy: 0.0249813 , rho : 1.1000E-6
    F[0] = F_x(strains,u,correct_force,volume0, x[0], x[1],t = 1)-F_x(strains,u,correct_force,volume0, 2450, 0.38,t = 1) #korrektur term ([E] = MPa, [F] = N) (Abweichung aufgrund des durchschnittlichen EL. vols)
    F[1] = F_x(strains,u,correct_force,volume0, x[0], x[1],t = 2)-F_x(strains,u,correct_force,volume0, 2450, 0.38,t = 2)
    
    E_iter.append(x[0]/1000)
    nu_iter.append(x[1])
    
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


fig,ax = plt.subplots(figsize=(10, 5))
ax.plot(E_iter,linestyle = "solid", linewidth = 4, color = "k",label = "E-Modul")
ax.hlines(2.45,0,len(E_iter)-1, color = "k", linestyle = "dashed")
ax.plot(nu_iter,linestyle = "solid", linewidth = 4, color = "g", label = r"$\nu$")
ax.hlines(0.38,0,len(E_iter)-1, color = "g", linestyle = "dashed")

ax.set_xlabel("Iteration",fontsize = 14)
ax.set_xticks(np.arange(len(E_iter)))
ax.grid()
ax.legend(fontsize = 12,loc = "center right")
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=8)


E = x[0]
nu = x[1]
d_eps = np.gradient(strains,axis = 0)[:,:,:]
du = np.gradient(u)
sig_y = []
plastic_strain = []
#d_eps : np.array, shape: (n_states,n_ip,6), zeitl. Ableitungen der Dehnungen an jedem Integrationspunkt, aller Zeitschritte

for t in range(strains.shape[0]-1):
    #bei einer Abweichung zwischen dW_in und dW_ex um > 2 J wird die approximierte Fließspannung bestimmt
    if abs(F_x(strains,u,correct_force,mean_el_volumes[t], E, nu,t)) > 2:
        plastic_strain.append(calc_effective_strain(np.mean(strains[t,:,:],axis = 0)))
        d_eps_I = calc_effective_strain(np.mean(d_eps[t,:,:],axis = 0))
        
        #direktes Auflösen der Fließspannung, nach der Energiebilanz, unter der Annahme, dass ein homogenes Spannungsfeld vorliegt
        sig_y.append(correct_force[t]*du[t]/(probe_volumes[t]*d_eps_I))
        

sig_y = np.array(sig_y)
plastic_strain = np.array(plastic_strain)
#abziehen der elastischen Dehnung (die Dehnung, die beim ersten mal vorliegt, wenn die Energiebilanz nicht mehr erfüllt wird)
#damit tatsächlich ausschließlich die plastische Dehnung vorliegt
plastic_strain = plastic_strain-plastic_strain[0]



sig_y_savgol = signal.savgol_filter(sig_y,21,1)

it_cy = np.concatenate((np.append(plastic_strain,1).reshape([-1,1]),np.append(sig_y_savgol,sig_y_savgol[-1]).reshape([-1,1])),axis = 1)


fig,ax = plt.subplots(figsize=(8, 8))
ax.plot(plastic_strain, sig_y, label = "Approximation", linestyle = "solid", linewidth = 4, color = "k")
ax.plot(plastic_strain, sig_y_savgol, label = "geglättete Approximation", linestyle = "dashed", linewidth = 4, color = "b")
ax.plot(cy[:100,0],cy[:100,1], label = "korrekte Fließkurve", linestyle = "solid", linewidth = 4, color = "g")

ax.legend(loc = "lower right",fontsize = 12)
ax.grid(visible = True)
ax.set_ylim(bottom = 0)
ax.set_xlabel("$\epsilon_{p}$ [-]",fontsize = 14)
ax.set_ylabel("$\sigma_{y}$ [MPa] ",fontsize = 14)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=8)

ax.set_title(titel,fontsize = 14)