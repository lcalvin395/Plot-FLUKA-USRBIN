import pylab as plt
import numpy as np

import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib import ticker
import statistics as st

particle1 ="elec"
f=0
g=0
q=[]
avenergy=[]
total=0

if particle1=='muon':
    # opening and creating new .txt file 
    with open( 
        "muon_production_42_plot.dat", 'r') as r, open( 
            'muon_production_42_plot2.txt', 'w') as o: 
        
        for line in r: 
            #strip() function 
            if line.strip(): 
                o.write(line)                                                  #STRIPPING OUT THE BLANK LINES IN THE ORIGNAL AND CREATING A NEW ONE
                                                                            #BLANK LINES WERE BEING READ AND ADDED AS DATA.....
    # opening and creating new .txt file 
    with open( 
        "muon_production_39_plot.dat", 'r') as r, open( 
            'muon_production_39_plot2.txt', 'w') as o: 
        
        for line in r: 
            #strip() function 
            if line.strip(): 
                o.write(line) 
    



    y, x, z, err = np.loadtxt('muon_production_42_plot2.txt', unpack=True)
    length = len(z)

    while g<len(z):
        if z[g]>0:
            #print("pixel number:",g)
            q.append(g)
        g=g+1




    particle ="energymuon"
    f=0
    g=0

    total=0
    y, x, z, err = np.loadtxt('muon_production_39_plot2.txt', unpack=True)


if particle1=='elec':
    # opening and creating new .txt file 
    with open( 
        "/Users/lukecalvin/2023/eli-np-collim_chamber_lead_box/2_mags_full_setup_extra_mag_and_mag_up_16/muon_production_40_plot.dat", 'r') as r, open( 
            '/Users/lukecalvin/2023/eli-np-collim_chamber_lead_box/2_mags_full_setup_extra_mag_and_mag_up_16/muon_production_40_plot2.txt', 'w') as o: 
        
        for line in r: 
            #strip() function 
            if line.strip(): 
                o.write(line)                                                  #STRIPPING OUT THE BLANK LINES IN THE ORIGNAL AND CREATING A NEW ONE
                                                                            #BLANK LINES WERE BEING READ AND ADDED AS DATA.....
    # opening and creating new .txt file 
    with open( 
        "/Users/lukecalvin/2023/eli-np-collim_chamber_lead_box/2_mags_full_setup_extra_mag_and_mag_up_16/muon_production_39_plot.dat", 'r') as r, open( 
            '/Users/lukecalvin/2023/eli-np-collim_chamber_lead_box/2_mags_full_setup_extra_mag_and_mag_up_16/muon_production_39_plot2.txt', 'w') as o: 
        
        for line in r: 
            #strip() function 
            if line.strip(): 
                o.write(line) 
    



    y, x, z, err = np.loadtxt('/Users/lukecalvin/2023/eli-np-collim_chamber_lead_box/2_mags_full_setup_extra_mag_and_mag_up_16/muon_production_40_plot2.txt', unpack=True)
    length = len(z)

    while g<len(z):
        if z[g]>0:
            #print("pixel number:",g)
            q.append(g)
        g=g+1




    particle ="energyelec"
    f=0
    g=0

    total=0
    y, x, z, err = np.loadtxt('/Users/lukecalvin/2023/eli-np-collim_chamber_lead_box/2_mags_full_setup_extra_mag_and_mag_up_16/muon_production_39_plot2.txt', unpack=True)



length = len(z)

print(len(z))
percentdone=[0,10,20,30,40,50,60,70,80,90]
checkedpixels=[]
perc=0

listofclustersize=[]

do='yes'

#normalising the energy colourbar and setting limit
if do=="yes":
    while g<len(z):
        if (round((g/len(z))*100))!=perc:
            perc=(round((g/len(z))*100))
            print(perc,'%')
        if (z[g]*18431863354*0.00546875*0.00546875*0.1)>0.1:
            
            z[g]=z[g]*0
        if (z[g]*18431863354*0.00546875*0.00546875*0.1)<0.0000001:
            
            z[g]=z[g]*0
        if (z[g]*18431863354*0.00546875*0.00546875*0.1)<0.1:
            z[g]=z[g]*18431863354*0.00546875*0.00546875*0.1
        if (g in q)==True:
            #print("muon pixel energy deposited:", z[g])
            avenergy.append(z[g])
        if (g in q)==False:
            z[g]=z[g]*0

        #true=(g==q)
        #print(true)
        #print((x))
        #print(g)
        n=0
        if (g not in checkedpixels):
            print(g)
        if z[g]>0 and ((g not in checkedpixels)):    #if pixel value is non zero and pixel hasnt been counted in a previous cluster
            prevpixel=[]
            pixel=g
            pixelist=[g]
            clusterpixels=[]
            #print('pixellist',pixelist)
            n=0
            e=0
            
            while len(pixelist)>0:
                #print('here')
                if ((pixelist[0] in prevpixel)==True):
                    pixelist.remove(pixelist[0])
                    continue
                
                pixel=pixelist[0]
                clusterpixels.append(pixel)
                n=n+1
                #print('N:',n)
                #print(pixelist)
                #print(prevpixel)
                #print(pixel)
                #print(pixel)
                check=0
                try:
                    if z[pixel+(256)]>0 and ((pixel+(256) in prevpixel)==False):
                        check=1
                        #print(check)
                        pixelist.append(pixel+(256))
                except:
                    print()
                
                try:
                    if z[pixel+(255)]>0 and ((pixel+(255) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel+(255))
                except:
                    print()
                try:
                    if z[pixel+(257)]>0 and ((pixel+(257) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel+(257))
                except:
                    print()
                try:
                    if z[pixel-(256)]>0 and ((pixel-(256) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-(256))
                except:
                    print()
                try:
                    if z[pixel-(255)]>0 and ((pixel-(255) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-(255))
                except:
                    print()
                try:
                    if z[pixel-(257)]>0 and ((pixel-(257) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-(257))
                except:
                    print()
                try:
                    if z[pixel+1]>0 and ((pixel+(1) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel+(1))
                except:
                    print()
                try:
                    if z[pixel-1]>0 and ((pixel-(1) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-1)
                except:
                    print()
                prevpixel.append(pixel)
                checkedpixels.append(pixel)
                pixelist.remove(pixel)
            if check==0:
                #print('N HERE:',n)
                #print(n)
                e=1       # CHECK IS ZERO so set e equal to 1 to indicate that for pixel g, all pixels in this cluster have been found. 
                if n>2000:
                    for k in range(0,len(clusterpixels)):
                        z[clusterpixels[k]]=0
        if n>0:
            listofclustersize.append(n)
        g=g+1
    meanavenergy = sum(avenergy)/len(avenergy)
    medavenergy=st.median(avenergy)
    maxavenergy=max(avenergy)
    minavenergy=min(avenergy)
    print("mean energy deposited of {}: {}".format(particle1,meanavenergy))
    print("median energy deposited of {}: {}".format(particle1,medavenergy))
    print("max energy deposited of {}: {}".format(particle1,maxavenergy))
    print("min energy deposited of {}: {}".format(particle1,minavenergy))

"""
while g<len(z):
    if z[g]>0:
        print("pixel number:",g)
        q.append(g)
    g=g+1
"""

"""
while f<len(z):
    if x[f]>0.83124998584 and x[f]<0.84218748566 and y[f]>16.322656162 and y[f]>16.344531156 and z[f]>0:
        print(x[f],y[f],z[f])
    f=f+1
"""
aveclustersize=sum(listofclustersize)/len(listofclustersize)
print('AVERAGE CLUSTER SIZE:',aveclustersize)   
print('CLUSTER SIZES:',len(listofclustersize))
    
plt.hist(listofclustersize)
plt.show()
# Sample data
side = np.linspace(-2,2,15)
#print(side)
X,Y = np.meshgrid(side,side)
Z = np.zeros((length,1))
while f < length:
    Z[f][0]=z[f]
    f=f+1
z=z
#print(Z)
#fig, ax = plt.subplots()
# Plot the density map using nearest-neighbor interpolation
#plt.pcolormesh(x,y,Z)
#plt.show()
#, norm=colors.LogNorm()

plt.hist2d(x, y,(255,255), weights=z)
cb = plt.colorbar()
tick_locator = ticker.MaxNLocator(nbins=10)
cb.locator = tick_locator
cb.update_ticks()
plt.savefig('{}_usrbin.png'.format(particle), bbox_inches='tight')
plt.show()

plt.hist2d(x, y,(255,255), weights=z, norm=colors.LogNorm())
plt.colorbar()


plt.savefig('{}_log_usrbin.png'.format(particle), bbox_inches='tight')
plt.show()
n=0
sum1 = 0
print(z)
print(len(z))
while n < len(z):
    if -1 <= y[n] <=1:
        sum1=sum1+z[n]
    n=n+1
print(total)