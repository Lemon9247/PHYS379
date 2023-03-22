#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import math
import pandas

with open("trimmed_data.csv",newline="") as file:
    data = csv.reader(file,delimiter=",")
    masses = [] #creates new list to store masses
    radii = [] #creates new list to store radii
    #data is given in units of jupiter for mass and radii
    names = []
    temperatures = [] #creates list to store temperatures (in kelvin)
    volumes = [] #creates list to store volumes
    densities = [] #creates list to store densities
    vesc = [] #creates list to store escape velocities
    G = 6.67*pow(10,-11) #gravitational constant (m^3*kg^-1*s^-2)
    indices = [] #creates list to store ESI
    
    
    for i,line in enumerate(data): #fills lists
        names.append(line[0])
        masses.append(line[1])
        radii.append(line[2])
        temperatures.append(line[3])
    
    for i in range(len(radii)): #calculation of volumes
        v = (4/3)*(math.pi)*pow((float(radii[i])*69911000),3) #multiplied by radius of jupiter (m)
        volumes.append(v)
    
    for i in range(len(radii)): #calculation of densities
        volume = volumes[i]
        mass = float(masses[i])*(1.898*pow(10,27)) #multiplied by mass of jupiter (kg)
        density = mass/volume
        densities.append(density) #densities will be in kg/m^3
        
    for i in range(len(radii)): #calculation of escape velocities
        m = float(masses[i])*(1.898*pow(10,27)) #multiplied by mass of jupiter (kg)
        r = float(radii[i])*69911000 #multiplied by radius of jupiter (m)
        velocity = math.sqrt((2*G*m)/r)
        vesc.append(velocity) #in (m/s)
    
    #Parameters required for calculation of ESI
    #weights:
    w_r = 0.57 #weight of radius
    w_d = 1.07 #weight of density
    w_v = 0.7 #weight of escape velocity
    w_t = 5.58 #weight of temperature
    
    for i in range(len(radii)):
        term1 = pow((1 - abs(((float(radii[i])*69911000)-6371000)/((float(radii[i])*69911000)+6371000))),w_r) #radius of earth=6371000m
        term2 = pow((1 - abs((float(densities[i])-5520)/(float(densities[i])+5520))),(w_d/2)) #density of earth=5520kg/m^3
        term3 = pow((1 - abs((float(vesc[i])-11186)/(float(vesc[i])+11186))),(w_v/3)) #vesc earth= 11186m/s
        term4 = pow((1 - abs((float(temperatures[i])-288)/(float(temperatures[i])+288))),(w_t/4)) #temp earth=288K
        esi = term1*term2*term3*term4
        indices.append(esi)
    
    df = pandas.DataFrame({'Name' : names, 'ESI' : indices})  
    df.to_csv('esi.csv', index=False, encoding='utf-8')

