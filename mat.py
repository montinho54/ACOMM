# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 13:34:06 2022

@author: monti
"""

# -*- coding: utf-8 -*-
import numpy as np

class Wrapper:
    def __init__(self,keyword_path,element_type):
        """
        Parameters
        ----------
        keyword_path : String
            Pfad, der Keyword-Datei in dem folgende Materialkonstanten enhalten sind:
            - e (E-Modul in der selben Einheit, wie die FLießspannung der Fließkurve (Mpa oder Gpa)), 
            - pr (Querkontraktionszahl), 
            - eppf (Float) effective plastic strain at which material softening begins (logaritmic strain), 
            - eppfr (Float) effective plastic strain at which material ruptures (logaritmic strain)
            Desweiteren muss die Keyword-Datei die Loadcurve enthalten, welche die Fließspannung über die effektive plastische Dehnung definiert.
            
        element_type : String
            shell, shl, Shell 
            oder:
            solid, sld, Solid

        Returns
        -------
        None.

        """
        self.keyword_path = keyword_path
        self.element_type = element_type
        
    def calc_stresses(self,strains,cm,**kwargs):
        """
        Ruft das um47shl Materialmodell als Wrapper auf
    
        Parameters
        ----------                                                  
        strains : np.array mit shape = (steps,n_ips,6)
            Dehnungspfad für jede Dehnung in jedem Integrationspunkt (eps_xx,eps_yy,eps_zz,eps_xy,eps_yz,eps_zx)
            Häufig wird nur für einen Integrationspunkt gerechnet. Dann hat strains folgende Shape: (steps,1,6)
        
        cm : Liste mit Material Koeffizienten [Emod,nue,crv/table,eppf,eppfr,y0 = Fließspannung]
            cm[0]                                   = Emod (E-Modul in Mpa)
            cm[1]                                   = nue (Querkontraktionszahl)
            cm[2]                                   = eppf (Float) effective plastic strain at which material softening begins (logaritmic strain)
            cm[3]                                   = eppfr (Float) effective plastic strain at which material ruptures (logaritmic strain)  
            cm[4]                                   = y0 (Fließspannung in Mpa)
            cm[5]                                   = mid (Material ID)
            cm[6]                                   = lcss (Loadcurve ID im Material)
              
        kwargs
        ------
        verbose : int
            verbose = 0: keine Ausgabe (default)
            verbose = 2: Ausgabe der verwendeten Materialkonstanten aus der Keyword-Datei 
                        + aktueller Status des Radial-Return Algorithmus innerhalb der Material-Subroutine 

        
            
        Parameter, die in Zukunft hinzugefügt werden können:
        -----------------------------------------------------
        dt1siz : float
            Zeitschrittgröße (hat aktuell noch keinen Einfluss auf das Ergebnis, würde die Dehnratenabhängigkeit beeinflussen) 
        crv/table : Integer
            Integer, der die passende Fließkurve bei Berücksichtigung der Dehnratenabhängigkeit bestimmt
            
    
        Returns
        -------
        stresses : np.array mit shape = (steps,n_ips,6)
            Array, das jede Spannung (in der Einheit, der Fließkurve und des E-Moduls (Mpa oder Gpa)) in jedem Integrationspunkt enthält 
            (sig_xx,sig_yy,sig_zz,sig_xy,sig_yz,sig_zx)
            Üblicherweise wird nur für einen Integrationspunkt gerechnet. Dann hat stresses folgende Shape: (steps,1,6).
        """
        #verbose
        if "verbose" in kwargs:
            verbose = kwargs["verbose"]
        else:
            verbose = 0
        
        #prüfen, ob nur ein dehnungszustand eines Integrationspunktes zu einem Zeitpunkt übergeben wurde (Single Point One Step)
        spos = False
        if len(strains.shape) == 1:
            spos = True
            strains = strains.reshape(1,1,strains.shape[0])
        

    
    
        steps = np.shape(strains)[0] #number of steps
        n_ips = np.shape(strains)[1] #number of integration points 
        sig1,sig2,sig3,sig4,sig5,sig6 = np.zeros(n_ips),np.zeros(n_ips),np.zeros(n_ips),np.zeros(n_ips),np.zeros(n_ips),np.zeros(n_ips)   
        tepsp = np.zeros(n_ips)
        d3 = np.zeros(n_ips)  
        hsvs = np.zeros([n_ips,3])     
        incs = np.zeros([steps,n_ips,6])
        stresses = np.zeros([steps,n_ips,6])
        
        
        for ip in range(n_ips):
            for eps in range(6):
                incs[0,ip,eps] = strains[0,ip,eps] #erstes Inkrement ist der erste Dehnungseintrag
                incs[1:,ip,eps] = np.array([j-i for i, j in zip(strains[:-1,ip,eps], strains[1:,ip,eps])])
        
        
        for i in range(len(incs)):                
            if self.element_type == "sld" or self.element_type == "solid" or self.element_type == "Solid":
                sig1,sig2,sig3,sig4,sig5,sig6,d3,tepsp,hsvs = mat_sld(cm,incs[i,:,0],incs[i,:,1],incs[i,:,2],incs[i,:,3],incs[i,:,4],incs[i,:,5],sig1,sig2,sig3,sig4,sig5,sig6,tepsp,hsvs,self.keyword_path,**kwargs)
            elif self.element_type == "shl" or self.element_type == "shell" or self.element_type == "Shell":
                sig1,sig2,sig3,sig4,sig5,sig6,d3,tepsp,hsvs = mat_shl(cm,incs[i,:,0],incs[i,:,1],incs[i,:,2],incs[i,:,3],incs[i,:,4],incs[i,:,5],sig1,sig2,sig3,sig4,sig5,sig6,tepsp,hsvs,self.keyword_path,**kwargs)
            else:
                print("mat81:\nEs wurde ein falscher Elementtyp bestimmt. Folgende Elementtypen sind möglich:\nshell, shl, Shell\noder:\nsolid, sld, Solid")
            stresses[i,:,0] = sig1
            stresses[i,:,1] = sig2
            stresses[i,:,2] = sig3
            stresses[i,:,3] = sig4
            stresses[i,:,4] = sig5
            stresses[i,:,5] = sig6
                
                 
        if spos:
            stresses = stresses.reshape(stresses.shape[2])
        return stresses


def mat_sld(cm,d1,d2,d3,d4,d5,d6,sig1,sig2,sig3,sig4,sig5,sig6,epsps,hsvs,keyword_path,**kwargs):
    """
    Materialmodell für Solids:
        - führt grundlegenen Spannungsberechnungen für Solids durch (v.Mises Plastizität -> Radial Return) 

    Parameters
    ----------                                                  
    cm : Liste mit Material Koeffizienten [Emod,nue,crv/table,eppf,eppfr,y0 = Fließspannung]
        cm[0]                                   = Emod (E-Modul in Mpa)
        cm[1]                                   = nue (Querkontraktionszahl)
        cm[2]                                   = eppf (Float) effective plastic strain at which material softening begins (logaritmic strain)
        cm[3]                                   = eppfr (Float) effective plastic strain at which material ruptures (logaritmic strain)  
        cm[4]                                   = y0 (Fließspannung in Mpa)
        cm[5]                                   = mid (Material ID)
        cm[6]                                   = lcss (Loadcurve ID im Material)
        
    d1,d2,d3,d4,d5,d6 : Floats
        Dehnungsinkremente der IP's [d_eps_xx],[d_eps_yy],[d_eps_zz],[d_eps_xy],[d_eps_yz],[d_eps_zx], shape = (n_ips,) (jeweils)
        
    sig1,sig2,sig3,sig4,sig5,sig6 : Floats
        Spannungen aus letzten Zeitschritt der IP's [sig_xx],[sig_yy],[sig_zz],[sig_xy],[sig_yz],[sig_zx], shape = (n_ips,) (jeweils)
        
    epsps : Float
        effective plastic Strain aus letztem Zeitschritt, shape = (n_ips,)
    
    hsvs : Liste mit History Variablen [dmg, plastic effective strainrate,total effective strain rate]
        hsvs[0]                                 = dmg, shape = (n_ips,)
        hsvs[1]                                 = plastic effective strain rate, shape = (n_ips,)
        hsvs[2]                                 = total effective strain rate, shape = (n_ips,)
        
        
    keyword_path : String
        absoluter Pfad (String) der Keyword-Datei
        
    mid : int
        Bestimmt welches Material (Material-ID) ausgewertet werden soll. Wenn kein mid übergeben wird, wird das erste Material in der Keyword-Datei ausgewertet 
        
    kwargs
    ------
    verbose : int
        verbose = 0: 
        verbose = 0: keine Ausgabe 
        verbose = 1: Ausgabe der verwendeten Materialkonstanten aus der Keyword-Datei (default)
        verbose = 2: Ausgabe der verwendeten Materialkonstanten aus der Keyword-Datei 
                    + aktueller Status des Radial-Return Algorithmus innerhalb der Material-Subroutine t)

        
    Parameter, die in Zukunft hinzugefügt werden können:
    ---------------------------------------------------
    dt1siz : float
        Zeitschrittgröße (hat aktuell noch keinen Einfluss auf das Ergebnis, würde die Dehnratenabhängigkeit beeinflussen) 
    crv/table : Integer
        Integer, der die passende Fließkurve bei Berücksichtigung der Dehnratenabhängigkeit bestimmt
        

    Returns
    -------
    sig1,sig2,sig3,sig4,sig5,sig6 [Mpa] : np.arrays
        Berechnete Spannungen im aktuellen Zeitschritt, shape = (n_ips,)
    
    d3 : np.array
        geupdatete Dehnung in Dickenrichtung, shape = (n_ips,)
        
    tepsp : Float
        geupdatete effective plastic strain, shape = (n_ips,)
    
    hsvs : np.array
        geupdatete history variablen
        hsvs[0]                                 = dmg, shape = (n_ips,)
        hsvs[1]                                 = plastic effective strain rate, shape = (n_ips,)
        hsvs[2]                                 = total effective strain rate, shape = (n_ips,)  
    """
    
    """own implementation"""
    
    nlq = len(d1) #nlq = anzahl der Integrationspunkte
       

    tepsp = np.zeros([nlq])
    
    tig1,tig2,tig3,tig4,tig5,tig6 = np.zeros([nlq]),np.zeros([nlq]),np.zeros([nlq]),np.zeros([nlq]),np.zeros([nlq]),np.zeros([nlq])
    q1,q2,q3,aj2,yield1,yield2 = np.zeros([nlq]),np.zeros([nlq]),np.zeros([nlq]),np.zeros([nlq]),np.zeros([nlq]),np.zeros([nlq])
    q1,q2,q3,aj2,yield1 = np.zeros([nlq]),np.zeros([nlq]),np.zeros([nlq]),np.zeros([nlq]),np.zeros([nlq])
    isav,sigy,hard,dmg = np.zeros([nlq]),np.zeros([nlq]),np.zeros([nlq]),np.zeros([nlq])
    
    
    failels = np.array(np.zeros(nlq), dtype=bool)
    #############################################################################################################################
    #verbose
    if "verbose" in kwargs:
        verbose = kwargs["verbose"]
    else:
        verbose = 1
        
    #Fortran-Berechnungen:
        
    #Material Constants
    emod = cm[0]
    rnue = cm[1]
    #crvid = cm[2]

    gmod =emod/(2.*(1.+rnue))
    bkmod=emod/(3.*(1.-2.*rnue))
    
    eppf=cm[2] #eppf = (effective plastic strain at which softening begins)
    eppfr=max(eppf,cm[3]) #cm[4] = eppfr (effective plastic strain at which rupture begins)
    frs=eppfr-eppf 
    
    #numerical parameters
    itmax=20 
    tol=1.e-6
    
    #remove damage
    for i in range(nlq):
        dmg[i] = hsvs[i,0]
        dmgi = 1/(1-dmg[i])
        sig1[i]=sig1[i]*dmgi
        sig2[i]=sig2[i]*dmgi
        sig3[i]=sig3[i]*dmgi
        sig4[i]=sig4[i]*dmgi
        sig5[i]=sig5[i]*dmgi
        sig6[i]=sig6[i]*dmgi
    

  
    #trial stresses and yield function

    for i in range(nlq):
        tepsp[i]=epsps[i] #!trail plastic strain to last plastic #jeder IP hat einen Wert für effective plastic strain
        epsvol=(d1[i]+d2[i]+d3[i])/3. #!get volumetric strain 
        dpress=3.0*bkmod*epsvol #!sigma increment due to volumetric
        tig1[i]=sig1[i]+dpress+2.0*gmod*(d1[i]-epsvol)
        tig2[i]=sig2[i]+dpress+2.0*gmod*(d2[i]-epsvol)
        tig3[i]=sig3[i]+dpress+2.0*gmod*(d3[i]-epsvol)            
        tig4[i]=sig4[i]+gmod*d4[i]
        tig5[i]=sig5[i]+gmod*d5[i]
        tig6[i]=sig6[i]+gmod*d6[i]
        press=1./3.*(tig1[i]+tig2[i]+tig3[i]) #!new pressure
        q1[i]=tig1[i]-press #!deviatroics
        q2[i]=tig2[i]-press
        q3[i]=tig3[i]-press
        #entspricht: J2 = 0.5*S:S
        aj2[i]=(tig4[i]**2+tig5[i]**2+tig6[i]**2-q1[i]*q2[i]-q2[i]*q3[i]-q1[i]*q3[i]) 
        

        #Ausgeben der Fließspannung in jedem IP
        sigy[i],hard[i] = get_sigy_hard(keyword_path,tepsp[i])
        
        #checken, ob die Fließbedingung verletzt wird (ist das Ergebnis <=0 oder >0)
        yield1[i] = np.sqrt(3.0*aj2[i])-sigy[i]
        
        #yield2 dient zum Print-out des ursprünglichen Ergebnisses, wenn Fließbedingung verletzt wird
        yield2[i] = np.sqrt(3.0*aj2[i])-sigy[i]
        
        
        
    #sort IPs ...
    j=0
    for i in range(nlq):
        #plastic: need return mapping below
        if yield1[i] > 0: 
            isav[j]=i  #isav speichert die Indizes (also die IP's) ab, in denen die Fließspannung überschritten worden ist
            j=j+1
            # elastic: stresses = trial stresses, done.
        else:
            if verbose == 2:
                print("tepsp: {:.7f} | d1: {:.3f} | d2: {:.3f} | d3: {:.8f} | (elastic strain -> no return mapping needed)".format(tepsp[i],d1[i],d2[i],d3[i]))
    

 
    """
    return mapping: Newton-Algorithmus, wird nur an den IP's aufgerufen, an denen yield1 > 0 ist, also Plastizität aufgetreten ist.
    Bei den restlichen IP's wird diese Schleife übersprungen (da j in "sort IPs.." nur für yield1>0 inkrementiert wird)
    """
    for ii in range(j):
      
        i=int(isav[ii])
        depi=0.0
        sqaj2=np.sqrt(3.0*aj2[i])

        for iter1 in range(itmax):
           
            #perform max. itmax iterations
 
            func=sqaj2-3.0*gmod*depi-sigy[i]
            fprm=-3.0*gmod-hard[i]
            depi=depi-func/(fprm+1.e-20)
            tepsp[i]=epsps[i]+depi

            #ausgeben der Fließspannung in jedem IP
            sigy[i],hard[i] = get_sigy_hard(keyword_path,tepsp[i])
            
            if verbose == 2:
                print("tepsp: {:.7f} | d1: {:.3f} | d2: {:.3f} | d3: {:.8f} | ".format(tepsp[i],d1[i],d2[i],d3[i]))
            
            yield1[i]=sqaj2-3.0*gmod*depi-sigy[i]
            check=abs(yield1[i])/sqaj2
            if check > tol and iter1 == itmax-1:
                if verbose == 2:
                    print("did not converge after {}( = itmax) iterations, check: {} tol: {}".format(itmax,check,tol))
                return
            elif check < tol:
                if verbose == 2:
                    print("converged after {} iterations, check: {} tol: {}\n".format(iter1+1,check,tol))
                break
            
            
        #stress and history update
        deps=3.0*gmod*depi/sqaj2
        tig1[i]=tig1[i]-deps*q1[i]
        tig2[i]=tig2[i]-deps*q2[i]
        tig3[i]=tig3[i]-deps*q3[i]
        tig4[i]=tig4[i]-deps*tig4[i]
        tig5[i]=tig5[i]-deps*tig5[i]
        tig6[i]=tig6[i]-deps*tig6[i]

                  
     
   #stress & strain update 
    for i in range(nlq):
        sig1[i]=tig1[i]
        sig2[i]=tig2[i]
        sig3[i]=tig3[i]
        sig4[i]=tig4[i]
        sig5[i]=tig5[i]
        sig6[i]=tig6[i]
        epsps[i]=tepsp[i]
        
    #compute damage
    for ii in range(j):
        i=int(isav[ii])
        if frs > 0.0: #ist der Fall, wenn eppfr > eppf ist
            dmg[i]=max(dmg[i],(epsps[i]-eppf)/frs)
        else:           
            #03.11. Fehler in der Implementierung
            #FORTRAN: SIGN(A,B) returns the value of A with the sign of B.
            #numpy: The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0. nan is returned for nan inputs.
            #dmg[i]=0.5+np.sign([0.5,epsps[i]-eppf])
            #aus diesem Grund wird die FORTRAN routine "sign" folgendermaßen umgangen
            
            dmg[i]=0.5+0.5*np.sign(epsps[i]-eppf)
        
        dmg[i]=min(0.99999,dmg[i])
        hsvs[i,0]=dmg[i] #hsv(1) = damage 
        
    #add damage
    for i in range(nlq):
        dmgi=1.-dmg[i]
        sig1[i]=sig1[i]*dmgi
        sig2[i]=sig2[i]*dmgi
        sig3[i]=sig3[i]*dmgi
        sig4[i]=sig4[i]*dmgi
        sig5[i]=sig5[i]*dmgi
        sig6[i]=sig6[i]*dmgi
   
    #failure
    for i in range(nlq):
        if (dmg[i] >= 0.99999): #wenn dmg > 0.999 ist, hat das Element versagt und wird gelöscht
            sig1[i]=0.0
            sig2[i]=0.0
            sig3[i]=0.0
            sig4[i]=0.0
            sig5[i]=0.0
            sig6[i]=0.0
            failels[i] = True
            
    return [sig1,sig2,sig3,sig4,sig5,sig6,d3,tepsp,hsvs]



def mat_shl(cm,d1,d2,d3,d4,d5,d6,sig1,sig2,sig3,sig4,sig5,sig6,epsps,hsvs,keyword_path,**kwargs):
    """
    Materialmodell für Shells:
        - führt grundlegenen Spannungsberechnungen für Schalen durch (v.Mises Plastizität -> Radial Return) 
        - Zugelassen ist nur der ebene Spannungszustand

    Parameters
    ----------                                                  
    cm : Liste mit Material Koeffizienten [Emod,nue,crv/table,eppf,eppfr,y0 = Fließspannung]
        cm[0]                                   = Emod (E-Modul in Mpa)
        cm[1]                                   = nue (Querkontraktionszahl)
        cm[2]                                   = eppf (Float) effective plastic strain at which material softening begins (logaritmic strain)
        cm[3]                                   = eppfr (Float) effective plastic strain at which material ruptures (logaritmic strain)  
        cm[4]                                   = y0 (Fließspannung in Mpa)
        cm[5]                                   = mid (Material ID)
        cm[6]                                   = lcss (Loadcurve ID im Material)
        
    d1,d2,d3,d4,d5,d6 : Floats
        Dehnungsinkremente der IP's [d_eps_xx],[d_eps_yy],[d_eps_zz],[d_eps_xy],[d_eps_yz],[d_eps_zx], shape = (n_ips,) (jeweils)
        
    sig1,sig2,sig3,sig4,sig5,sig6 : Floats
        Spannungen aus letzten Zeitschritt der IP's [sig_xx],[sig_yy],[sig_zz],[sig_xy],[sig_yz],[sig_zx], shape = (n_ips,) (jeweils)
        
    epsps : Float
        effective plastic Strain aus letztem Zeitschritt, shape = (n_ips,)
    
    hsvs : Liste mit History Variablen [dmg, plastic effective strainrate,total effective strain rate]
        hsvs[0]                                 = dmg, shape = (n_ips,)
        hsvs[1]                                 = plastic effective strain rate, shape = (n_ips,)
        hsvs[2]                                 = total effective strain rate, shape = (n_ips,)
        
        
    keyword_path : String
        absoluter Pfad (String) der Keyword-Datei
        
    mid : int
        Bestimmt welches Material (Material-ID) ausgewertet werden soll. Wenn kein mid übergeben wird, wird das erste Material in der Keyword-Datei ausgewertet 
        
    kwargs
    ------
    verbose : int
        verbose = 0: 
        verbose = 0: keine Ausgabe 
        verbose = 1: Ausgabe der verwendeten Materialkonstanten aus der Keyword-Datei (default)
        verbose = 2: Ausgabe der verwendeten Materialkonstanten aus der Keyword-Datei 
                    + aktueller Status des Radial-Return Algorithmus innerhalb der Material-Subroutine 

        
    Parameter, die in Zukunft hinzugefügt werden können:
    ---------------------------------------------------
    dt1siz : float
        Zeitschrittgröße (hat aktuell noch keinen Einfluss auf das Ergebnis, würde die Dehnratenabhängigkeit beeinflussen) 
    crv/table : Integer
        Integer, der die passende Fließkurve bei Berücksichtigung der Dehnratenabhängigkeit bestimmt
        

    Returns
    -------
    sig1,sig2,sig3,sig4,sig5,sig6 [Mpa] : np.arrays
        Berechnete Spannungen im aktuellen Zeitschritt, shape = (n_ips,)
    
    d3 : np.array
        geupdatete Dehnung in Dickenrichtung, shape = (n_ips,)
        
    tepsp : Float
        geupdatete effective plastic strain, shape = (n_ips,)
    
    hsvs : np.array
        geupdatete history variablen
        hsvs[0]                                 = dmg, shape = (n_ips,)
        hsvs[1]                                 = plastic effective strain rate, shape = (n_ips,)
        hsvs[2]                                 = total effective strain rate, shape = (n_ips,)  
    """
    
    """own implementation"""
    
    nlq = len(d1) #nlq = anzahl der Integrationspunkte
       
    shsig = np.zeros([nlq,2])
    shdeps = np.zeros([nlq,2])
    deps3 = np.zeros([nlq])
    tepsp = np.zeros([nlq])
    
    tig1,tig2,tig3,tig4,tig5,tig6 = np.zeros([nlq]),np.zeros([nlq]),np.zeros([nlq]),np.zeros([nlq]),np.zeros([nlq]),np.zeros([nlq])
    q1,q2,q3,aj2,yield1,yield2 = np.zeros([nlq]),np.zeros([nlq]),np.zeros([nlq]),np.zeros([nlq]),np.zeros([nlq]),np.zeros([nlq])
    q1,q2,q3,aj2,yield1 = np.zeros([nlq]),np.zeros([nlq]),np.zeros([nlq]),np.zeros([nlq]),np.zeros([nlq])
    isav,sigy,hard,dmg = np.zeros([nlq]),np.zeros([nlq]),np.zeros([nlq]),np.zeros([nlq])
    
    deps3 = np.zeros([nlq])
    shsig = np.zeros([nlq,2])
    shdeps = np.zeros([nlq,2])
    
    failels = np.array(np.zeros(nlq), dtype=bool)
    #############################################################################################################################
    #verbose
    if "verbose" in kwargs:
        verbose = kwargs["verbose"]
    else:
        verbose = 1
        
    #Fortran-Berechnungen:
        
    #Material Constants
    emod = cm[0]
    rnue = cm[1]
    #crvid = cm[2]

    gmod =emod/(2.*(1.+rnue))
    bkmod=emod/(3.*(1.-2.*rnue))
    
    eppf=cm[2] #eppf = (effective plastic strain at which softening begins)
    eppfr=max(eppf,cm[3]) #cm[4] = eppfr (effective plastic strain at which rupture begins)
    frs=eppfr-eppf 
    
    #numerical parameters
    itmax=20 
    tol=1.e-6
    
    #remove damage
    for i in range(nlq):
        dmg[i] = hsvs[i,0]
        dmgi = 1/(1-dmg[i])
        sig1[i]=sig1[i]*dmgi
        sig2[i]=sig2[i]*dmgi
        sig3[i]=sig3[i]*dmgi
        sig4[i]=sig4[i]*dmgi
        sig5[i]=sig5[i]*dmgi
        sig6[i]=sig6[i]*dmgi
    
    #initialize plane stress iteration for shells
    #jcount=1
    
    for i in range(nlq):
        shsig[i,0]=0
        shsig[i,1]=0
        shdeps[i,0]=0
        shdeps[i,1]=0
        deps3[i]=-rnue/(1.-rnue)*(d1[i]+d2[i])
        
    for jcount in range(2,5):  
        #trial stresses and yield function
    
        for i in range(nlq):
            tepsp[i]=epsps[i] #!trail plastic strain to last plastic #jeder IP hat einen Wert für effective plastic strain
            epsvol=(d1[i]+d2[i]+deps3[i])/3. #!get volumetric strain 
            dpress=3.0*bkmod*epsvol #!sigma increment due to volumetric
            tig1[i]=sig1[i]+dpress+2.0*gmod*(d1[i]-epsvol)
            tig2[i]=sig2[i]+dpress+2.0*gmod*(d2[i]-epsvol)
            tig3[i]=sig3[i]+dpress+2.0*gmod*(deps3[i]-epsvol)            
            tig4[i]=sig4[i]+gmod*d4[i]
            tig5[i]=sig5[i]+gmod*d5[i]
            tig6[i]=sig6[i]+gmod*d6[i]
            press=1./3.*(tig1[i]+tig2[i]+tig3[i]) #!new pressure
            q1[i]=tig1[i]-press #!deviatroics
            q2[i]=tig2[i]-press
            q3[i]=tig3[i]-press
            #entspricht: J2 = 0.5*S:S
            aj2[i]=(tig4[i]**2+tig5[i]**2+tig6[i]**2-q1[i]*q2[i]-q2[i]*q3[i]-q1[i]*q3[i]) 
            

            #Ausgeben der Fließspannung in jedem IP
            sigy[i],hard[i] = get_sigy_hard(keyword_path,tepsp[i])
            
            #checken, ob die Fließbedingung verletzt wird (ist das Ergebnis <=0 oder >0)
            yield1[i] = np.sqrt(3.0*aj2[i])-sigy[i]
            
            #yield2 dient zum Print-out des ursprünglichen Ergebnisses, wenn Fließbedingung verletzt wird
            yield2[i] = np.sqrt(3.0*aj2[i])-sigy[i]
            
   
            
        #sort IPs ...
        j=0
        for i in range(nlq):
            #plastic: need return mapping below
            if yield1[i] > 0: 
                isav[j]=i  #isav speichert die Indizes (also die IP's) ab, in denen die Fließspannung überschritten worden ist
                j=j+1
                # elastic: stresses = trial stresses, done.
            else:
                if verbose == 2:
                    print("jcount: {} | tepsp: {:.7f} | d1: {:.3f} | d2: {:.3f} | deps3: {:.8f} | (elastic strain -> no return mapping needed)".format(jcount,tepsp[i],d1[i],d2[i],deps3[i]))
        
    
     
        """
        return mapping: Newton-Algorithmus, wird nur an den IP's aufgerufen, an denen yield1 > 0 ist, also Plastizität aufgetreten ist.
        Bei den restlichen IP's wird diese Schleife übersprungen (da j in "sort IPs.." nur für yield1>0 inkrementiert wird)
        """
        for ii in range(j):
          
            i=int(isav[ii])
            depi=0.0
            sqaj2=np.sqrt(3.0*aj2[i])
    
            for iter1 in range(itmax):
               
                #perform max. itmax iterations
     
                func=sqaj2-3.0*gmod*depi-sigy[i]
                fprm=-3.0*gmod-hard[i]
                depi=depi-func/(fprm+1.e-20)
                tepsp[i]=epsps[i]+depi

                #ausgeben der Fließspannung in jedem IP
                sigy[i],hard[i] = get_sigy_hard(keyword_path,tepsp[i])
                
                if verbose == 2:
                    print("jcount: {} | tepsp: {:.7f} | d1: {:.3f} | d2: {:.3f} | deps3: {:.8f} |".format(jcount,tepsp[i],d1[i],d2[i],deps3[i]))
                
                yield1[i]=sqaj2-3.0*gmod*depi-sigy[i]
                check=abs(yield1[i])/sqaj2
                if check > tol and iter1 == itmax-1:
                    if verbose == 2:
                        print("did not converge after {}( = itmax) iterations, check: {} tol: {}".format(itmax,check,tol))
                    return
                elif check < tol:
                    if verbose == 2:
                        print("converged after {} iterations, check: {} tol: {}\n".format(iter1+1,check,tol))
                    break
                
                
            #stress and history update
            deps=3.0*gmod*depi/sqaj2
            tig1[i]=tig1[i]-deps*q1[i]
            tig2[i]=tig2[i]-deps*q2[i]
            tig3[i]=tig3[i]-deps*q3[i]
            tig4[i]=tig4[i]-deps*tig4[i]
            tig5[i]=tig5[i]-deps*tig5[i]
            tig6[i]=tig6[i]-deps*tig6[i]

                
   
        #update of deps3 for plane stress iteration 
        for i in range(nlq):

            shdeps[i,0]=shdeps[i,1] #shdeps = np.zeros([nlq,2])
            shdeps[i,1]=deps3[i]
            shsig[i,0]=shsig[i,1]
            shsig[i,1]=tig3[i]
                
        if jcount == 2:
            for i in range(nlq):
                deps3[i]=-(d1[i]+d2[i])
            
            
        if jcount == 3:
            for i in range(nlq):
                ddeps=shdeps[i,1]-shdeps[i,0]
                dsig=shsig[i,1]-shsig[i,0]
                afac=0.0
                if (abs(dsig) >= 1.e-14):
                    afac=ddeps/dsig
                deps3[i]=shdeps[i,1]-shsig[i,1]*afac
        
        if jcount == 4:
            break

   
     
   #stress & strain update 
    for i in range(nlq):
        d3[i]=deps3[i]
        sig1[i]=tig1[i]
        sig2[i]=tig2[i]
        sig3[i]=0.0
        sig4[i]=tig4[i]
        sig5[i]=tig5[i]
        sig6[i]=tig6[i]
        epsps[i]=tepsp[i]
        
    #compute damage
    for ii in range(j):
        i=int(isav[ii])
        if frs > 0.0: #ist der Fall, wenn eppfr > eppf ist
            dmg[i]=max(dmg[i],(epsps[i]-eppf)/frs)
        else:
            #03.11. Fehler in der Implementierung
            #FORTRAN: SIGN(A,B) returns the value of A with the sign of B.
            #numpy: The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0. nan is returned for nan inputs.
            #dmg[i]=0.5+np.sign([0.5,epsps[i]-eppf])
            #aus diesem Grund wird die FORTRAN routine "sign" folgendermaßen umgangen
            
            dmg[i]=0.5+0.5*np.sign(epsps[i]-eppf)
        
        dmg[i]=min(0.99999,dmg[i])
        hsvs[i,0]=dmg[i] #hsv(1) = damage 
        
    #add damage
    for i in range(nlq):
        dmgi=1.-dmg[i]
        sig1[i]=sig1[i]*dmgi
        sig2[i]=sig2[i]*dmgi
        sig3[i]=sig3[i]*dmgi
        sig4[i]=sig4[i]*dmgi
        sig5[i]=sig5[i]*dmgi
        sig6[i]=sig6[i]*dmgi
   
    #failure
    for i in range(nlq):
        if (dmg[i] >= 0.99999): #wenn dmg > 0.999 ist, hat das Element versagt und wird gelöscht
            sig1[i]=0.0
            sig2[i]=0.0
            sig3[i]=0.0
            sig4[i]=0.0
            sig5[i]=0.0
            sig6[i]=0.0
            failels[i] = True
            
    return [sig1,sig2,sig3,sig4,sig5,sig6,d3,tepsp,hsvs]

def read_lc_from_keyword(keyword_path,**kwargs):
    
    """
    Liest eine Keyword-Datei ein und gibt die darin enthaltene Loadcurve als numpy-array aus
     
    Parameters
    ----------
    keyword_path : string
        Pfad der Keyword-Datei
        
    kwargs
    ------
    first_line_to_read : int
        Legt die erste Zeile fest, ab der eingelesen werden soll    
    sfa : int 
        Scale Faktor der Abszisse
    sfo : int
        Scale Faktor der Ordinate
    offa : int
        Offset Faktor der Abszisse
    offo : int
        Offset Faktor der Ordinate
     
    Returns
    -------
    lc : np.array (shape: n,2)
        np.array der ermittelten Loadcurve
    """
    
    if "sfa" in kwargs:
        sfa = kwargs["sfa"]
    else:
        sfa = 1
    if "sfo" in kwargs:
        sfo = kwargs["sfo"]
    else:
        sfo = 1
    if "offa" in kwargs:
        offa = kwargs["offa"]
    else:
        offa = 0
    if "offo" in kwargs:
        offo = kwargs["offo"]
    else:
        offo = 0
    
    
    
    with open(keyword_path) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
        
        if "first_line_to_read" in kwargs:
            first_line = kwargs["first_line_to_read"]
            lines = lines[first_line:]
        
    for i,line in enumerate(lines):
        if "lcid" in line:
            start_line = i+3
            break
           
    for j,line in enumerate(lines[i:]):
        if line == "*END":
            end_line = j+i
            break
            
    lc = lines[start_line:end_line] 
    lc_list = []
    for line in lc:
        if "," in line:
            try:
                i = line.find(",")
                a = float(line[:i])
                o = float(line[i+1:])
                lc_list.append([sfa*a+offa,sfo*o+offo])
            except:
                break
            
        else:
            try:
                a = float(line[:20])
                o = float(line[20:])
                lc_list.append([sfa*a+offa,sfo*o+offo])
            except:
                break

    lc_arr = np.array(lc_list)         
    return lc_arr


               
            
        

def crvval(crv,xval):
    """
    Extrapoliert die vorgegebene Fließkurve linear anhand der letzten zwei Stützstellen 
    (In LS-Dyna wird die Fließkurve standardmäßig linear extrapoliert)
    Führt eine lineare Interpolation von crv durch
    und ermittelt den Rückgabewert an xval


    Parameters
    ----------
    crv : np.array() ,shape: n,2
        Loadcurve, erste Spalte: Effektive plastische Dehnungn, zweite Spalte: effektive Fließspannung
    xval : float
        Effektive plastische Dehnung, an der die effektive Fließspannung ermittelt werden soll

    Returns
    -------
    yval : float
        Effektive Fließspannung an xval
    slope : float
    
        Steigung der Loadcurve an xval (wird durch zentrale Differenzen und anschließender linearen Interpolation ermittelt)
    """
    

    #lineare Extrapolation der Fließkurve
    fit = np.polyfit(crv[-2:,0],crv[-2:,1],1)
    x = 20
    line = np.poly1d(fit)
    y = line(x)
    crv = np.vstack([crv, [x,y]])


    yval = np.interp(xval, crv[:,0], crv[:,1])
    #crv[:,0] is nicht äquidistant -> dx variiert
    grad = np.gradient(crv[:,1],crv[:,0])
    slope = np.interp(xval, crv[:,0], grad)
    
    return yval, slope   
 

def search_lc_with_lcid(keyword_path,lcss):
    """
    Ließt die Keyword-Datei in Keywordpath ein und gibt die Zeile der Loadcurve aus, welche die Loadcurve-ID = lcss besitzt.

    Parameters
    ----------
    keyword_path : string
        pfad der Keyword Datei, welche die Materialparameter enthält
    lcss : int
        Loadcurve-ID, der Loadcurve, die im geünschten Material hinterlegt ist

    Returns
    -------
    first_line : int
        Zeile, ab der die Loadcurve beginnt, die der Loadcurve-ID entspricht

    """
    with open(keyword_path) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
        
        
        for i,line in enumerate(lines):
            if line == "*DEFINE_CURVE_TITLE":
                j = 3
                break
            elif line == "*DEFINE_CURVE":
                j = 2
                break
        s = i+j
        line = lines[s]
    
        for j in range(1,11):

            if line[10-j-1] == " ":      
                lcid = int(line[10-j:10])
                break
            
        while lcid != lcss:
            
            for i,line in enumerate(lines):
                if line == "*DEFINE_CURVE_TITLE" and i > s:
                    j = 3
                    break
                elif line == "*DEFINE_CURVE" and i > s:
                    j = 2
                    break
            s = i+j
            line = lines[s]
        
            for j in range(1,11):

                if line[10-j-1] == " ":      
                    lcid = int(line[10-j:10])
                    break
    first_line = s-1
    return first_line
                
    

def get_sigy_hard(keyword_path,xval):
    """
    Eigene implementierung der get_sigy_hard subroutine
  
    Parameters
    ----------
    keyword_path : string
        Pfad der Keyword-Datei 
    xval : float
        Effektive plastische Dehnung an der die Fließspannung ausgewertet werden soll

    Returns
    -------
    yval : float
        FLießspannung (sigy) 
    slope : float
        Steigung der loadcurve an xval (im FORTRAN Code: hard[i])

    """
    #ursprüngliche Version
    """
    mid = int(cm[5])

    mid,i = read_material_constants(keyword_path, mid = mid)
    cm_dict = read_constants_from_keyword(keyword_path,first_constant = "mid", n_lines = 2, first_line_to_read = i)
    lcid = cm_dict["lcss"]
    i = search_lc_with_lcid(keyword_path,lcid)
    consts = read_constants_from_keyword(keyword_path,first_constant = "lcid", n_lines = 1,first_line_to_read = i)

    lcid = int(consts["lcid"])
    sfa = consts["sfa"]
    sfo = consts["sfo"]
    offa = consts["offa"]
    offo = consts["offo"]

    crv= read_lc_from_keyword(keyword_path,first_line_to_read = i, sfa = sfa, sfo = sfo, offa = offa, offo = offo)
    yval,slope = crvval(crv,xval)
    """
    yval = 0.02
    slope = 0.0
    return yval, slope          