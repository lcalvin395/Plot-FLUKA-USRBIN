import matplotlib.pyplot as plt
import numpy as np

import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib import ticker
import statistics as st


#MATPLOTLIB INTERACTIVE MODE TURNED OFF (FOR PLOTS)#

plt.ioff()

particle1 ="elec"
f=0
g=0
q=[]
avenergy=[]
plotmodeaveenergy=[]
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
meanaveenergyincluster=[]
modeaveenergyincluster=[]
maxenergyincluster=[]
totalenergyincluster=[]
coeffvariation=[]

primaries=18131863354
if particle1=='muon':
    primaries=250000

do='yes'

#normalising the energy colourbar and setting limit
if do=="yes":
    while g<len(z):
        if (round((g/len(z))*100))!=perc:
            perc=(round((g/len(z))*100))
            print(perc,'%')
        '''if (z[g]*18431863354*0.00546875*0.00546875*0.1)>0.1:
            
            z[g]=z[g]*0'''
        '''if (z[g]*18431863354*0.00546875*0.00546875*0.1)<0.0000001:
            
            z[g]=z[g]*0'''
        '''if (z[g]*primaries*0.00546875*0.00546875*0.1)>0:
            z[g]=z[g]*primaries*0.00546875*0.00546875*0.1'''
        if (g in q)==True:
            #print("muon pixel energy deposited:", z[g])
            avenergy.append(z[g]*primaries*0.00546875*0.00546875*0.1*(10**6))
            
        if (g in q)==False:
            z[g]=z[g]*0

        #true=(g==q)
        #print(true)
        #print((x))
        #print(g)   
        n=0
        
        if z[g]>0 and ((g not in checkedpixels)):    #if pixel value is non zero and pixel hasnt been counted in a previous cluster
            prevpixel=[]
            pixel=g
            pixelist=[g]
            clusterpixels=[]
            #print('pixellist',pixelist)
            n=0
            e=0
            clusterenergy=[]
            modeclusterenergy=[]
            while len(pixelist)>0:
                #print('here')
                if ((pixelist[0] in prevpixel)==True):
                    pixelist.remove(pixelist[0])
                    continue
                
                pixel=pixelist[0]
                clusterpixels.append(pixel)
                n=n+1
                clusterenergy.append(z[pixel])
                modeclusterenergy.append(z[pixel]*primaries*0.00546875*0.00546875*0.1*(10**6))
                if g==10679:
                    print(pixelist)
                    print('HERE FOR EACH PIXEL')
                    print(g)
                    print(pixel,x[pixel],y[pixel])
                #print('N:',n)
                #print(pixelist)
                #print(prevpixel)
                #print(pixel)
                #print(pixel)
                check=0

                try:
                    if z[pixel+(257)]>0 and ((pixel+(257) in prevpixel)==False):
                        check=1
                        #print(check)
                        pixelist.append(pixel+(257))
                        print('+257')
                except:
                    print()
                
                try:
                    if z[pixel+(256)]>0 and ((pixel+(256) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel+(256))
                        print('+256')
                except:
                    print()
                try:
                    if z[pixel+(258)]>0 and ((pixel+(258) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel+(258))
                        print('+258')
                        print(z[pixel+(258)])
                except:
                    print()
                try:
                    if z[pixel-(257)]>0 and ((pixel-(257) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-(257))
                        print('-257')
                except:
                    print()
                try:
                    if z[pixel-(256)]>0 and ((pixel-(256) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-(256))
                        print('-256')
                except:
                    print()
                try:
                    if z[pixel-(258)]>0 and ((pixel-(258) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-(258))
                        print('-258')
                except:
                    print()
                try:
                    if z[pixel+1]>0 and ((pixel+(1) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel+(1))
                        print('+1')
                except:
                    print()
                try:
                    if z[pixel-1]>0 and ((pixel-(1) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-1)
                        print('-1')
                except:
                    print()
                prevpixel.append(pixel)
                checkedpixels.append(pixel)
                pixelist.remove(pixel)
            if check==0:
                #print('N HERE:',n)
                #print(n)
                e=1       # CHECK IS ZERO so set e equal to 1 to indicate that for pixel g, all pixels in this cluster have been found. 
                if n>20000:
                    for k in range(0,len(clusterpixels)):
                        z[clusterpixels[k]]=0
        if n>0:
            meanaveenergyincluster.append(np.mean(clusterenergy))
            roundformode1=((np.array(modeclusterenergy))/5)
            roundformode2=np.around(roundformode1,0)
            roundformode=list((roundformode2)*5)
            roundformode.sort()
            roundformode=roundformode/max(roundformode)
            #print('LOOK HERE:')
            #print(modeclusterenergy)
            #print(roundformode)
            #print(st.mode(roundformode))
            modeaveenergyincluster.append(st.mode(roundformode))
            maxenergyincluster.append(max(clusterenergy))
            totalenergyincluster.append(sum(modeclusterenergy))
            if n>1:
                sigma=st.stdev(modeclusterenergy)
                emean=np.mean(modeclusterenergy)
                coeffvariation.append(sigma/emean)
            plotmodeaveenergy.append(st.mode(avenergy))
            #print('CLUSTER ENERGY:',clusterenergy)
            listofclustersize.append(n)
        #if n>19 and n<21:
            #print('HEREEEEEEEEEEEEE')
            #print(roundformode)
            #plt.plot(roundformode,'x-')
            
            #print(x[g],y[g])
        g=g+1
        avenergy=[]
    #meanavenergy = sum(avenergy)/len(avenergy)
    #modeavenergy=st.mode(avenergy)
    #maxavenergy=max(avenergy)
    #minavenergy=min(avenergy)
    #print("mean energy deposited of {}: {}".format(particle1,meanavenergy))
    #print("mode energy deposited of {}: {}".format(particle1,modeavenergy))
    #print("max energy deposited of {}: {}".format(particle1,maxavenergy))
    #print("min energy deposited of {}: {}".format(particle1,minavenergy))




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
        "/Users/lukecalvin/2023/eli-np-collim_chamber_lead_box/2_mags_full_setup_extra_mag_13/muon_production_40_plot.dat", 'r') as r, open( 
            '/Users/lukecalvin/2023/eli-np-collim_chamber_lead_box/2_mags_full_setup_extra_mag_13/muon_production_40_plot2.txt', 'w') as o: 
        
        for line in r: 
            #strip() function 
            if line.strip(): 
                o.write(line)                                                  #STRIPPING OUT THE BLANK LINES IN THE ORIGNAL AND CREATING A NEW ONE
                                                                            #BLANK LINES WERE BEING READ AND ADDED AS DATA.....
    # opening and creating new .txt file 
    with open( 
        "/Users/lukecalvin/2023/eli-np-collim_chamber_lead_box/2_mags_full_setup_extra_mag_13/muon_production_39_plot.dat", 'r') as r, open( 
            '/Users/lukecalvin/2023/eli-np-collim_chamber_lead_box/2_mags_full_setup_extra_mag_13/muon_production_39_plot2.txt', 'w') as o: 
        
        for line in r: 
            #strip() function 
            if line.strip(): 
                o.write(line) 
    



    y, x, z, err = np.loadtxt('/Users/lukecalvin/2023/eli-np-collim_chamber_lead_box/2_mags_full_setup_extra_mag_13/muon_production_40_plot2.txt', unpack=True)
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
    y, x, z, err = np.loadtxt('/Users/lukecalvin/2023/eli-np-collim_chamber_lead_box/2_mags_full_setup_extra_mag_13/muon_production_39_plot2.txt', unpack=True)



length = len(z)

print(len(z))
percentdone=[0,10,20,30,40,50,60,70,80,90]
checkedpixels=[]
perc=0








primaries=18417332000
if particle1=='muon':
    primaries=250000

do='yes'

#normalising the energy colourbar and setting limit
if do=="yes":
    while g<len(z):
        if (round((g/len(z))*100))!=perc:
            perc=(round((g/len(z))*100))
            print(perc,'%')
        '''if (z[g]*18431863354*0.00546875*0.00546875*0.1)>0.1:
            
            z[g]=z[g]*0'''
        '''if (z[g]*18431863354*0.00546875*0.00546875*0.1)<0.0000001:
            
            z[g]=z[g]*0'''
        '''if (z[g]*primaries*0.00546875*0.00546875*0.1)>0:
            z[g]=z[g]*primaries*0.00546875*0.00546875*0.1'''
        if (g in q)==True:
            #print("muon pixel energy deposited:", z[g])
            avenergy.append(z[g]*primaries*0.00546875*0.00546875*0.1*(10**6))
            
        if (g in q)==False:
            z[g]=z[g]*0

        #true=(g==q)
        #print(true)
        #print((x))
        #print(g)   
        n=0
        
        if z[g]>0 and ((g not in checkedpixels)):    #if pixel value is non zero and pixel hasnt been counted in a previous cluster
            prevpixel=[]
            pixel=g
            pixelist=[g]
            clusterpixels=[]
            #print('pixellist',pixelist)
            n=0
            e=0
            clusterenergy=[]
            modeclusterenergy=[]
            while len(pixelist)>0:
                #print('here')
                if ((pixelist[0] in prevpixel)==True):
                    pixelist.remove(pixelist[0])
                    continue
                
                pixel=pixelist[0]
                clusterpixels.append(pixel)
                n=n+1
                clusterenergy.append(z[pixel])
                modeclusterenergy.append(z[pixel]*primaries*0.00546875*0.00546875*0.1*(10**6))
                if g==10679:
                    print(pixelist)
                    print('HERE FOR EACH PIXEL')
                    print(g)
                    print(pixel,x[pixel],y[pixel])
                #print('N:',n)
                #print(pixelist)
                #print(prevpixel)
                #print(pixel)
                #print(pixel)
                check=0

                try:
                    if z[pixel+(257)]>0 and ((pixel+(257) in prevpixel)==False):
                        check=1
                        #print(check)
                        pixelist.append(pixel+(257))
                        print('+257')
                except:
                    print()
                
                try:
                    if z[pixel+(256)]>0 and ((pixel+(256) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel+(256))
                        print('+256')
                except:
                    print()
                try:
                    if z[pixel+(258)]>0 and ((pixel+(258) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel+(258))
                        print('+258')
                        print(z[pixel+(258)])
                except:
                    print()
                try:
                    if z[pixel-(257)]>0 and ((pixel-(257) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-(257))
                        print('-257')
                except:
                    print()
                try:
                    if z[pixel-(256)]>0 and ((pixel-(256) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-(256))
                        print('-256')
                except:
                    print()
                try:
                    if z[pixel-(258)]>0 and ((pixel-(258) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-(258))
                        print('-258')
                except:
                    print()
                try:
                    if z[pixel+1]>0 and ((pixel+(1) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel+(1))
                        print('+1')
                except:
                    print()
                try:
                    if z[pixel-1]>0 and ((pixel-(1) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-1)
                        print('-1')
                except:
                    print()
                prevpixel.append(pixel)
                checkedpixels.append(pixel)
                pixelist.remove(pixel)
            if check==0:
                #print('N HERE:',n)
                #print(n)
                e=1       # CHECK IS ZERO so set e equal to 1 to indicate that for pixel g, all pixels in this cluster have been found. 
                if n>20000:
                    for k in range(0,len(clusterpixels)):
                        z[clusterpixels[k]]=0
        if n>0:
            meanaveenergyincluster.append(np.mean(clusterenergy))
            roundformode1=((np.array(modeclusterenergy))/5)
            roundformode2=np.around(roundformode1,0)
            roundformode=list((roundformode2)*5)
            roundformode.sort()
            roundformode=roundformode/max(roundformode)
            #print('LOOK HERE:')
            #print(modeclusterenergy)
            #print(roundformode)
            #print(st.mode(roundformode))
            modeaveenergyincluster.append(st.mode(roundformode))
            maxenergyincluster.append(max(clusterenergy))
            totalenergyincluster.append(sum(modeclusterenergy))
            if n>1:
                sigma=st.stdev(modeclusterenergy)
                emean=np.mean(modeclusterenergy)
                coeffvariation.append(sigma/emean)
            plotmodeaveenergy.append(st.mode(avenergy))
            #print('CLUSTER ENERGY:',clusterenergy)
            listofclustersize.append(n)
        #if n>19 and n<21:
            #print('HEREEEEEEEEEEEEE')
            #print(roundformode)
            #plt.plot(roundformode,'x-')
            
            #print(x[g],y[g])
        g=g+1
        avenergy=[]
    #meanavenergy = sum(avenergy)/len(avenergy)
    #modeavenergy=st.mode(avenergy)
    #maxavenergy=max(avenergy)
    #minavenergy=min(avenergy)
    #print("mean energy deposited of {}: {}".format(particle1,meanavenergy))
    #print("mode energy deposited of {}: {}".format(particle1,modeavenergy))
    #print("max energy deposited of {}: {}".format(particle1,maxavenergy))
    #print("min energy deposited of {}: {}".format(particle1,minavenergy))




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
modeclustersize=st.mode(listofclustersize)
print('MODE CLUSTER SIZE: ', modeclustersize)

####################
plotclustersize=[]
plotaveenergyincluster=[]
intplotmaxenergyincluster=[]
plotmaxenergyincluster=[]

for a in range(1,100):
    intplotaveenergyincluster=[]
    for n in range(0,len(listofclustersize)):
        if a==listofclustersize[n]:
            
            intplotaveenergyincluster.append(meanaveenergyincluster[n])
    plotclustersize.append(a)
    plotaveenergyincluster.append((np.mean(intplotaveenergyincluster))*(10**6))

fig, ax=plt.subplots()   
ax.plot(plotclustersize,plotaveenergyincluster,'x')
plt.xlabel("Size of Cluster - pixels")
plt.ylabel('Mean Energy - KeV')
ax.set_xlim([0,40])                                                            #cluster size and MEAN ave energy
#ax.set_ylim([0,20])
plt.savefig('{}_clustersizes_and_mean_energy.png'.format(particle1), bbox_inches='tight')
#plt.show()

####################
print('mean energy of 6 pixel cluster :',(plotaveenergyincluster[5]))


####################
plotclustersize=[]
plotaveenergyincluster=[]
intplotmaxenergyincluster=[]
plotmaxenergyincluster=[]

for a in range(1,100):
    intplotaveenergyincluster=[]
    for n in range(0,len(listofclustersize)):
        if a==listofclustersize[n]:
            
            intplotaveenergyincluster.append(modeaveenergyincluster[n])
    plotclustersize.append(a)
    plotaveenergyincluster.append((np.mean(intplotaveenergyincluster)))

fig, ax=plt.subplots()   
ax.plot(plotclustersize,plotaveenergyincluster,'x')
plt.xlabel("Size of Cluster - pixels")
plt.ylabel('Mode Energy of Cluster - KeV')
ax.set_xlim([0,40])                                                            #cluster size and MODE ave energy
#ax.set_ylim([0,20])
plt.savefig('{}_clustersizes_and_mode_energy.png'.format(particle1), bbox_inches='tight')
#plt.show()


print('mode energy of 6 pixel cluster :',(plotaveenergyincluster[5]))
####################


####################
plotclustersize=[]
plotaveenergyincluster=[]
intplotmaxenergyincluster=[]
plotmaxenergyincluster=[]

for a in range(1,100):
    intplotmaxenergyincluster=[]
    for n in range(0,len(listofclustersize)):
        if a==listofclustersize[n]:
            
            intplotmaxenergyincluster.append(maxenergyincluster[n])
    plotclustersize.append(a)
    plotmaxenergyincluster.append((np.mean(intplotmaxenergyincluster))*(10**6))

fig, ax=plt.subplots()   
ax.plot(plotclustersize,plotmaxenergyincluster,'x')
plt.xlabel("Size of Cluster - pixels")
plt.ylabel('Max Energy - KeV')
ax.set_xlim([0,40])                                                            #cluster size and MAX energy
#ax.set_ylim([0,20])
plt.savefig('{}_clustersizes_and_max_energy.png'.format(particle1), bbox_inches='tight')
#plt.show()

print('MAX ENERGY:',(plotmaxenergyincluster[5]) )
####################

####################
plotclustersize=[]
plotaveenergyincluster=[]
intplotmaxenergyincluster=[]
plottotenergyincluster=[]

for a in range(1,100):
    intplottotenergyincluster=[]
    for n in range(0,len(listofclustersize)):
        if a==listofclustersize[n]:
            
            intplottotenergyincluster.append(totalenergyincluster[n])
    plotclustersize.append(a)
    plottotenergyincluster.append((np.mean(intplottotenergyincluster)))

fig, ax=plt.subplots()   
ax.plot(plotclustersize,plottotenergyincluster,'x')
plt.xlabel("Size of Cluster - pixels")
plt.ylabel('Total Energy - KeV')
ax.set_xlim([0,40])                                                            #cluster size and TOTAL energy
#ax.set_ylim([0,20])
plt.savefig('{}_clustersizes_and_total_energy.png'.format(particle1), bbox_inches='tight')
#plt.show()

print('TOTAL ENERGY:',(plottotenergyincluster) )
####################


weights=[]

for i in range(0,len(listofclustersize)):

    weights.append(1/len(listofclustersize))  
fig, ax=plt.subplots()   
binwidth=1
plt.hist(listofclustersize, bins=range(int(min(listofclustersize)), int(max(listofclustersize) + binwidth)), width=1, weights=weights)
plt.xlabel("Size of Cluster - pixels")
plt.ylabel('N')
xlim=40
ax.set_xlim([0,xlim])   
plt.xticks(np.arange(0, xlim, step=5))
                                                     #number of cluster sizes
#ax.set_ylim([0,20])
plt.savefig('/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV/{}_clustersizes.png'.format(particle1), bbox_inches='tight')
#plt.show()
# Sample data
fig, ax=plt.subplots()
ax = plt.gca()
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

plt.hist2d(x, y,(257,257), weights=z)
plt.xlabel("cm")
plt.ylabel('cm')
cb = plt.colorbar()
tick_locator = ticker.MaxNLocator(nbins=10)
cb.locator = tick_locator
cb.update_ticks()
plt.savefig('{}_usrbin.png'.format(particle1), bbox_inches='tight')
#plt.show()

#plt.hist2d(x, y,(257,257), weights=z, norm=colors.LogNorm())

#plt.colorbar()


#plt.savefig('{}_log_usrbin.png'.format(particle1), bbox_inches='tight')
#plt.show()
n=0
sum1 = 0
#print(z)
#print(len(z))
while n < len(z):
    if -1 <= y[n] <=1:
        sum1=sum1+z[n]
    n=n+1
#print(total)




g=100                       ############################################################          
#print(g)
#print(x[g],y[g],z[g])
g=g+257                     #proving that you need to do +257 for pixel above current pixel           
#print(g)
#print(x[g],y[g],z[g])       ############################################################ 
#


weights=[]

for i in range(0,len(totalenergyincluster)):

    weights.append(1/len(totalenergyincluster))  
binwidth=10
fig, ax=plt.subplots()
ax = plt.gca()
#ax.set_xlim([0,2000])
plt.xlabel("Total Energy in Cluster - KeV")
plt.ylabel('Probability')
#ax.hist(totalenergyincluster, bins=150, weights=weights)
ax.hist(totalenergyincluster, bins=range(int(min(totalenergyincluster)), int(max(totalenergyincluster) + binwidth), binwidth),weights=weights)
plt.savefig('/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV/total_energy_in_cluster_hist.png',bbox_inches='tight', dpi=1000)




weights=[]

for i in range(0,len(plotmodeaveenergy)):

    weights.append(1/len(plotmodeaveenergy))  
binwidth=10
fig, ax=plt.subplots()
ax = plt.gca()
ax.set_xlim([0,120])
plt.xlabel("Mode Energy in Cluster - KeV")
plt.ylabel('Probability')

#ax.hist(totalenergyincluster, bins=150, weights=weights)
ax.hist(plotmodeaveenergy, bins=range(int(min(plotmodeaveenergy)), int(max(plotmodeaveenergy) + binwidth), binwidth), color='b')
counts, bins, bars = ax.hist(plotmodeaveenergy, bins=range(int(min(plotmodeaveenergy)), int(max(plotmodeaveenergy) + binwidth), binwidth),weights=weights)
plt.savefig('/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV/{}_mode_energy_in_cluster_hist.png'.format(particle1),bbox_inches='tight', dpi=1000)
print(counts, bins, bars)


print(len(plotmodeaveenergy))
print(len(listofclustersize))


######
weights=[]

for i in range(0,len(coeffvariation)):

    weights.append(1/len(coeffvariation))  
binwidth=0.1
fig, ax=plt.subplots()
ax = plt.gca()
#ax.set_ylim([0,110])
plt.xlabel("Coefficient of Variation")
plt.ylabel('N')

#ax.hist(totalenergyincluster, bins=150, weights=weights)
#ax.hist(coeffvariation, bins=range(int(min(coeffvariation)), int(max(coeffvariation) + binwidth), binwidth), color='b')
ax.hist(coeffvariation, bins=(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5), color='b',weights=weights)
counts, bins, bars = ax.hist(coeffvariation, bins=(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5), color='b',weights=weights)
plt.savefig('/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV/{}_coeff_variation_in_cluster_hist.png'.format(particle1),bbox_inches='tight', dpi=1000)




#print(counts, bins, bars)
coeffsave=[[],[]]
for q in range(0,len(counts)):
    coeffsave[0].append(counts[q])
    coeffsave[1].append(bins[q])

print(coeffsave)
with open("/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV/eleccoeffsave.csv", "w") as txt_file:
    for q in range(0,len(counts)):
        txt_file.write("%g\t%g\n"%(float(coeffsave[1][q]),float(coeffsave[0][q]))) # works with any number of elements in a line
        

plt.show()