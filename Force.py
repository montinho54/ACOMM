# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 22:28:34 2021

@author: monti
"""

import numpy as np
import pandas as pd


def calc_dW_in(sigma,d_eps,ip_volumes):
    """
    Berechnet die differentielle innere Energie (Formänderungsenergie dW_in) im Körper, 
    basierend auf folgender Gleichung: dW_in = volumen_integral(d_eps:sigma)

    Parameters
    ----------
    sigma : np.array(), shape: (n_ip,6)
        Spannungen im aktuellen Zeitschritt (xx,yy,zz,xy,yz,zx)
    d_eps : np.array(), shape: (n_ip,6)
        Differentielle Dehnungen im aktuellen Zeitschritt 
        (wird über zentrale Differenzen aus vorherigen und nächsten Zeitschritt realisiert) (xx,yy,zz,xy,yz,zx)     
    ip_volumes : pd.Dataframe
        berechnet mittels:  
            force.calculate_el_volume(self.node_indexes,node_displacement,eps,self.t)
        shape:
            (n_elements,6) bzw. (n_elements,5)
        columns: 
            wenn rechtecks-elemente vorliegen: id1,id2,id3,id4,volume,area
            wenn dreiecks-elemente vorliegen: id1,id2,id3,volume,area

    Returns
    -------
    dW_in : float
        Differentielle innere Energie (Formänderungsenergie)

    """
  
        
    dW_in = 0
    for i in range(np.shape(sigma)[0]):
        
        strain_line = d_eps[i,:]
        stress_line = sigma[i,:]
        
        #doppelte Gewichtung der Einträge, welche nicht auf der Hauptdiagonalen des Spannungstensors liegen (xy,yz,zx),
        #damit die Tensorkntraktion korrekt durch das Skalarprodukt abgebildet wird 
        
        stress_line[3:] = 2*stress_line[3:]
        
        #sigma:d_eps = Tensorkontraktion, wird mit Skalarprodukt über jede Zeile realisiert
        p = np.dot(strain_line,stress_line)   
  
        dW_in = dW_in + ip_volumes.loc[i,"volume"]*p

    return dW_in
    
    

class Node:
    def __init__(self,ident,x,y,z):
        self.ident = ident
        self.x = x
        self.y = y
        self.z = z
       
def calculate_volume(element_shell_node_indexes,node_coord,shell_strains_step,t):
    """
    Berechnet das Volumen des gesamten Simulationskörpers indem jedes Elementvolumen berechnet wird
    und anschließend über alle Elementvolumina aufsummiert wird

    Parameters
    ----------
    element_shell_node_indexes : np.array , shape: (n_ip, 4)
        Array, in dem zu jedem Integrationspunkt die zugehörigen Elementknoten abgelegt sind
    node_coord : np.array(), shape: (n_nodes,3)
        Beinhaltet die x-,y-, und z- Koordinate jedes Element-Knotens im aktuellen Zeitschritt
        Notwendig zur Berechnung des geupdateten Element-Volumens (ip_volumes)
    shell_strains_step : np.array(), shape: (n_ip, 6)
        Beinhaltet die Dehnungen (xx_yy_zz_xy_yz_xz) aller Integrationspunkte im aktuellen Zeitschritt
    t : float
        Dicke des ursprünglichen Elements (im unbelasteten Zustand)
        Zur Berechnung des geupdateten Elementvolumens.

    Returns
    -------
    volume : float
        Volumen des Simulationskörpers


    """
    #t kann auch über d3plot ermittelt werden:
    #d3plot.arrays["element_shell_thickness"] #shape : (n_timesteps,n_elements)
    
    #tri mesh
    if element_shell_node_indexes.shape[1] == 3:
        names = ['id1', 'id2', 'id3']
        mesh = "tri"
    #quad mesh
    elif element_shell_node_indexes.shape[1] == 4:
        names = ['id1', 'id2', 'id3','id4']
        mesh = "quad"
        
    else:
        raise ValueError('Keine Shell-Elemente vorhanden - Volumenberechnung basiert auf Sehll-ELementen')
        
    node_indexes = pd.DataFrame(element_shell_node_indexes,columns = names)
    el_areas = []
    el_volumes = []
    thicknesses = []
    if mesh == "tri":
        for i in range(len(node_indexes)): #i geht über alle elemente
            nodes = []
            for name in names:
                idx = node_indexes.loc[i,name] #idx ist der Knotenindex
                nodes.append(Node(idx,node_coord[idx,0],node_coord[idx,1],node_coord[idx,2]))
            
            nod1 = nodes[0]
            nod2 = nodes[1]
            nod3 = nodes[2]
            #Oberflächenberechnung eines beliebigen Dreiecks
            #berechnet die Fläche eines Dreiecks mittels den Koordinaten in der Ebene, da von einer flachen Oberfläche ausgegangen wird
            #Formeln stammen aus: https://de.wikipedia.org/wiki/Dreiecksfläche ("Mit Koordinaten in der Ebene)
            a = 0.5*np.abs((nod2.x-nod1.x)*(nod3.y-nod1.y)-(nod3.x-nod1.x)*(nod2.y-nod1.y))
            
            #bestimmen der aktuellen Elementdicke, über z-Dehnungen,i = elementindex
            #zz_strain = np.exp(shell_strains_step[i,2])-1
            #thickness = zz_strain*t+t
            
            #Update 04.03.: es hat sich gezeigt, dass LS-Dyna die Elementvolumen mit einer Konstanten Elementdicke berechnet
            #unabhängig davon ob istupd = 0 oder istupd = 1 gewählt wird
            #mit einer Konstanten elementdicke wird die Innere energie am letzten Zeitschritt exakt getroffen
            
            thickness = t
            
            #Fläche mal Elementdicke
            el_volume = a*thickness
            el_areas.append(a)
            el_volumes.append(el_volume)
            thicknesses.append(thickness)
    elif mesh == "quad":
        for i in range(len(node_indexes)): #i geht über alle elemente
            nodes = []
            for name in names:
                idx = node_indexes.loc[i,name] #idx ist der Knotenindex
                nodes.append(Node(idx,node_coord[idx,0],node_coord[idx,1],node_coord[idx,2]))
            
            nod1 = nodes[0]
            nod2 = nodes[1]
            nod3 = nodes[2]
            nod4 = nodes[3]
            #Oberflächenberechnung eines beliebigen Rechtecks
            #A=(1/2)|[(x3-x1)(y4-y2) +(x4-x2)(y1-y3)]|
            #Quelle: http://www.mathematische-basteleien.de/ (Flächeninhalt-allgemeines Viereck)
            #Formel gilt nur unter der Bedingung, dass die Knoten gegen den Uhrzeigersinn indiziert sind (ist in LS-Dyna der Fall)
            
            a = 0.5*np.abs((nod3.x-nod1.x)*(nod4.y-nod2.y)+(nod4.x-nod2.x)*(nod1.y-nod3.y))
            
            #bestimmen der aktuellen Elementdicke, über z-Dehnungen,i = elementindex
            #zz_strain = np.exp(shell_strains_step[i,2])-1
            #thickness = zz_strain*t+t
            
            #Update 04.03.: es hat sich gezeigt, dass LS-Dyna die Elementvolumen mit einer Konstanten Elementdicke berechnet
            #unabhängig davon ob istupd = 0 oder istupd = 1 gewählt wird
            #mit einer Konstanten elementdicke wird die Innere energie am letzten Zeitschritt exakt getroffen
            
            thickness = t
              
            el_volume = a * thickness
            el_areas.append(a)
            el_volumes.append(el_volume)
            thicknesses.append(thickness)
            
            
    volume = sum(el_volumes)    
    return volume
    
    


def displacement(node_coordinates,node_displacement,boundary = "x"):
    """
    Berechnet die Verschiebung des äußersten Knoten, an dem die Kraft angreift

    Parameters
    ----------
    node_coordinates : np.array(), shape: (n_nodes,3)
        Beinhaltet die x-,y-, und z- Koordinate jedes Element-Knotens im unbelasteten Zustand,
        notwendig zur Ermittlung der äußeren Knoten
   
    node_displacement : node_coord : np.array(), shape: (steps,n_nodes,3)
        Beinhaltet die x-,y-, und z- Koordinate jedes Element-Knotens in allen Zeitschritten
        
    step : int
        Zeitschritt, an der die Verschiebung ausgegeben werden soll

    boundary : char (entweder "x" oder "y"), optional
        Rand an dem die Knoten zur Berechnung der Verschiebung liegen
        x -> vertikaler Rand
        y -> horizontaler Rand
        The default is "x".

    Returns
    -------
    disp : np.array, dType = float, shape: (n_timesteps)
        Verschiebung des äußersten Knotens über alle Zeitschritte

    """
     
    if boundary == "x":
        j = 0
    elif boundary == "y":
        j = 1
    else:
        raise ValueError("Attribut displacement_dir = {} ist ein unzulässiger Wert\nzugelassen sind: x bzw. y".format(boundary))
        
    
    outer_node_id = np.argsort(node_coordinates[:,j])[-1]
    outer_coord = node_coordinates[outer_node_id,j]
    disp = node_displacement[:,outer_node_id,j]-outer_coord
       
    
    return disp
        
        


