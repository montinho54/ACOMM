from lasso.dyna import D3plot, ArrayType,Binout
import numpy as np
import matplotlib.pyplot as plt

import os
import mat
import Force as force


#material = "T45"
material = "T65"

#Laden der Validierungssimulaitonen
pfad = os.path.abspath(os.getcwd()+ "/../Simulations/MAT81_" + material + "_shl")


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

u= force.displacement(node_coordinates,node_displacement,boundary = displacement_dir)
du = np.gradient(u)


stresses = d3plot.arrays[ArrayType.element_shell_stress][:,:,0,:]           
#: shape (n_states, n_shells_non_rigid, n_shell_layers, xx_yy_zz_xy_yz_xz) -> stresses unterscheiden seich nicht in z dimension, da ESZ)
strains = d3plot.arrays[ArrayType.element_shell_strain][:,:,0,:]            
#: shape (n_states, n_shells_non_rigid, upper_lower, xx_yy_zz_xy_yz_xz)  -> strains unterscheiden seich nicht in z dimension, da ESZ)

t0 = d3plot.arrays[ArrayType.element_shell_thickness][0]
node_coord_0 = node_displacement[0,:,:]
eps_0 = strains[0,:,:]
node_indexes = d3plot.arrays[ArrayType.element_shell_node_indexes]

#volumen des gesamtenprobekörpers
volume0 = force.calculate_volume(node_indexes,node_coord_0,eps_0,t0)


"""
stresses = d3plot.arrays[ArrayType.element_solid_stress]
stresses = stresses[:,:,0,:]
#: shape: (n_states, n_solids, n_solid_layers, xx_yy_zz_xy_yz_xz) -> n_solid_layers = 1
strains = d3plot.arrays[ArrayType.element_solid_strain][:,:,0,:]            
#: shape (n_states, n_shells_non_rigid, upper_lower, xx_yy_zz_xy_yz_xz)  -> n_solid_layers = 1
"""

#Single-element-Simulation

cm = [2.1,0.38,10e03,10e03,1,1,1]
sigmas = mat_shl.calc_stresses(strains,cm,verbose = 2)*1000
#sigmas = mat_sld.calc_stresses(strains,verbose = False)*1000




element = 0

eps_xx = strains[:,element,0]
sig_xx = sigmas[:,element,0]

#Plotten
fig,ax = plt.subplots(1,1,figsize = (10,10))
ax.scatter(strains[:,element,0],stresses[:,element,0]*1000,label = "Single element simulation (MAT81-Bayblend {})".format(material),c = "r",linewidths = 2,edgecolor = "k")
ax.set_xlabel("$\epsilon_{xy}$",fontsize = 15)
ax.set_ylabel("$\sigma_{xy} [MPa]$",fontsize = 15)

#Berechnen der Differenz:
diff = sig_xx-stresses[:,element,0]*1000
max_diff = max(np.abs(diff))


#Plotten
ax.plot(eps_xx,sig_xx,label = "Userdefined Material Model (MAT81-Bayblend {}) \napplying strain increments on updated stresses, $\epsilon_{{zz}}$ and $\epsilon_{{pl}}$".format(material),c = "b",linewidth = 2)
ax.set_title("Matmodell-Verification, eppf=1e012, eppfr = 1e014 \n max. difference: {:.5f} Mpa".format(max_diff),fontsize = 15)
ax.legend(fontsize = 15,loc = 'lower right' )
ax.grid()
fig.tight_layout()
