import pylab as plt
import numpy as np
from numpy import savetxt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib import ticker
import statistics as st
import pandas as pd
import math
from scipy.optimize import curve_fit



def myf(x, A, B): # this is your 'straight line' y=f(x)
    return A*x +B

class Solution: 
   def solve(self, clusterxy): 
        (x0, y0), (xlast, ylast) = clusterxy[0], clusterxy[len(clusterxy)-1] 
        x=[]
        y=[]
        try:
            mainslope=float(ylast-y0)/float(xlast-x0)
            mainangle=math.atan(mainslope)*(180/math.pi)
        except:
            if ylast-y0==0:
                mainslope=0
                mainangle=0
            if xlast-x0==0:
                mainslope=float('inf')
                mainangle=90
        print(mainangle)
        for i in range(0,len(clusterxy)):
            x.append(clusterxy[i][0])
            y.append(clusterxy[i][1])
        print(x)
        print(y)
        popt, pcov = curve_fit(myf, x, y)
        
        poptangle=math.atan(popt[0])*(180/math.pi)
        print(poptangle)
        #print(np.sqrt(np.diag(pcov)))
        print('\n')
        if mainslope==0:
            i=int(len(y)/2)
            if y[i]!=ylast:
                return False
                
            if y[i]==ylast:
                return True
                    
        if mainslope==float('inf'):
            i=int(len(x)/2)
            if x[i]!=xlast:
                return False
                
            if x[i]==xlast:
                return True
        if (mainslope!=float('inf')) and (mainslope!=0):
            #print(abs((popt[0]-mainslope)/mainslope))
            if abs(mainangle-poptangle)>5:
                return False
            else:
                return True
ob = Solution()


#MATPLOTLIB INTERACTIVE MODE TURNED OFF (FOR PLOTS)#
plt.ioff()
jp=0
particle1 ="muon"
f=0
g=0
q=[]
avenergy=[]
total=0
here=[]
xy=[]
xytf=[]
straight=[]
if particle1=='muon':
    # opening and creating new .txt file 
    with open( 
        "/Users/lukecalvin/2023/eli_np_muon_primaries_0.7GeV/muon_production_42_plot.dat", 'r') as r, open( 
            '/Users/lukecalvin/2023/eli_np_muon_primaries_0.7GeV/muon_production_42_plot2.txt', 'w') as o: 
        
        for line in r: 
            #strip() function 
            if line.strip(): 
                o.write(line)                                                  #STRIPPING OUT THE BLANK LINES IN THE ORIGNAL AND CREATING A NEW ONE
                                                                            #BLANK LINES WERE BEING READ AND ADDED AS DATA.....
    # opening and creating new .txt file 
    with open( 
        "/Users/lukecalvin/2023/eli_np_muon_primaries_0.7GeV/muon_production_39_plot.dat", 'r') as r, open( 
            '/Users/lukecalvin/2023/eli_np_muon_primaries_0.7GeV/muon_production_39_plot2.txt', 'w') as o: 
        
        for line in r: 
            #strip() function 
            if line.strip(): 
                o.write(line) 
    



    y, x, z, err = np.loadtxt('/Users/lukecalvin/2023/eli_np_muon_primaries_0.7GeV/muon_production_42_plot2.txt', unpack=True)
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
    y, x, z, err = np.loadtxt('/Users/lukecalvin/2023/eli_np_muon_primaries_0.7GeV/muon_production_39_plot2.txt', unpack=True)


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
meanenergyincluster=[]
coeffvariation=[]
plotmodeaveenergy=[]


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
            clusterxy=[]
            while len(pixelist)>0:
                #print('here')
                if ((pixelist[0] in prevpixel)==True):
                    pixelist.remove(pixelist[0])
                    continue
                
                pixel=pixelist[0]
                clusterpixels.append(pixel)
                n=n+1
                clusterxy.append([x[pixel],y[pixel]])
                clusterenergy.append(z[pixel])
                modeclusterenergy.append(z[pixel]*primaries*0.00546875*0.00546875*0.1*(10**6))
                '''if g==10679:
                    print(pixelist)
                    print('HERE FOR EACH PIXEL')
                    print(g)
                    print(pixel,x[pixel],y[pixel])'''
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
                        #print('+257')
                except:
                    print()
                
                try:
                    if z[pixel+(256)]>0 and ((pixel+(256) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel+(256))
                        #print('+256')
                except:
                    print()
                try:
                    if z[pixel+(258)]>0 and ((pixel+(258) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel+(258))
                        #print('+258')
                        #print(z[pixel+(258)])
                except:
                    print()
                try:
                    if z[pixel-(257)]>0 and ((pixel-(257) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-(257))
                        #print('-257')
                except:
                    print()
                try:
                    if z[pixel-(256)]>0 and ((pixel-(256) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-(256))
                        #print('-256')
                except:
                    print()
                try:
                    if z[pixel-(258)]>0 and ((pixel-(258) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-(258))
                        #print('-258')
                except:
                    print()
                try:
                    if z[pixel+1]>0 and ((pixel+(1) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel+(1))
                        #print('+1')
                except:
                    print()
                try:
                    if z[pixel-1]>0 and ((pixel-(1) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-1)
                        #print('-1')
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
                if ob.solve(clusterxy)==True:
                    sigma=st.stdev(modeclusterenergy)
                    emean=np.mean(modeclusterenergy)
                    coeffvariation.append(sigma)
                    listofclustersize.append(n)
                    int1=((np.array(avenergy))/5)
                    int2=np.around(int1,0)
                    int3=list((int2)*5)
                    plotmodeaveenergy.append(st.mode(int3))
                    meanenergyincluster.append(np.mean(modeclusterenergy))

                    here.append(modeclusterenergy)

                print(ob.solve(clusterxy))
                if ob.solve(clusterxy)==False:
                    straight.append(0)
                if ob.solve(clusterxy)==True:
                    straight.append(1)
            if n==1:
                listofclustersize.append(n)
                int1=((np.array(avenergy))/5)
                int2=np.around(int1,0)
                int3=list((int2)*5)
                plotmodeaveenergy.append(st.mode(int3))


                #print('n:',n)
                #print(xytf)
                #print('n: ',n)
            #print('CLUSTER ENERGY:',clusterenergy)
        #if n>19 and n<21:
            #print('HEREEEEEEEEEEEEE')
            #print(roundformode)
            #plt.plot(roundformode,'x-')
            
            #print(x[g],y[g])
        g=g+1
        avenergy=[]
    '''meanavenergy = sum(avenergy)/len(avenergy)
    modeavenergy=st.mode(avenergy)
    maxavenergy=max(avenergy)
    minavenergy=min(avenergy)
    print("mean energy deposited of {}: {}".format(particle1,meanavenergy))
    print("mode energy deposited of {}: {}".format(particle1,modeavenergy))
    print("max energy deposited of {}: {}".format(particle1,maxavenergy))
    print("min energy deposited of {}: {}".format(particle1,minavenergy))'''

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



fig, ax=plt.subplots()   
plt.hist(listofclustersize,bins=(len(listofclustersize)))
plt.xlabel("Size of Cluster - pixels")
plt.ylabel('N')
ax.set_xlim([0,40])                                                        #number of cluster sizes
#ax.set_ylim([0,20])
plt.savefig('{}_clustersizes.png'.format(particle1), bbox_inches='tight')
#plt.show()
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




particle1 ="muon"
f=0
g=0
q=[]
avenergy=[]
total=0

if particle1=='muon':
    # opening and creating new .txt file 
    with open( 
        "/Users/lukecalvin/2023/eli_np_muon_primaries_0.5GeV/muon_production_42_plot.dat", 'r') as r, open( 
            '/Users/lukecalvin/2023/eli_np_muon_primaries_0.5GeV/muon_production_42_plot2.txt', 'w') as o: 
        
        for line in r: 
            #strip() function 
            if line.strip(): 
                o.write(line)                                                  #STRIPPING OUT THE BLANK LINES IN THE ORIGNAL AND CREATING A NEW ONE
                                                                            #BLANK LINES WERE BEING READ AND ADDED AS DATA.....
    # opening and creating new .txt file 
    with open( 
        "/Users/lukecalvin/2023/eli_np_muon_primaries_0.5GeV/muon_production_39_plot.dat", 'r') as r, open( 
            '/Users/lukecalvin/2023/eli_np_muon_primaries_0.5GeV/muon_production_39_plot2.txt', 'w') as o: 
        
        for line in r: 
            #strip() function 
            if line.strip(): 
                o.write(line) 
    



    y, x, z, err = np.loadtxt('/Users/lukecalvin/2023/eli_np_muon_primaries_0.5GeV/muon_production_42_plot2.txt', unpack=True)
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
    y, x, z, err = np.loadtxt('/Users/lukecalvin/2023/eli_np_muon_primaries_0.5GeV/muon_production_39_plot2.txt', unpack=True)


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


meanaveenergyincluster=[]
modeaveenergyincluster=[]
maxenergyincluster=[]


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
                '''if g==10679:
                    print(pixelist)
                    print('HERE FOR EACH PIXEL')
                    print(g)
                    print(pixel,x[pixel],y[pixel])'''
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
                        #print('+257')
                except:
                    print()
                
                try:
                    if z[pixel+(256)]>0 and ((pixel+(256) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel+(256))
                        #print('+256')
                except:
                    print()
                try:
                    if z[pixel+(258)]>0 and ((pixel+(258) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel+(258))
                        #print('+258')
                        #print(z[pixel+(258)])
                except:
                    print()
                try:
                    if z[pixel-(257)]>0 and ((pixel-(257) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-(257))
                        #print('-257')
                except:
                    print()
                try:
                    if z[pixel-(256)]>0 and ((pixel-(256) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-(256))
                        #print('-256')
                except:
                    print()
                try:
                    if z[pixel-(258)]>0 and ((pixel-(258) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-(258))
                        #print('-258')
                except:
                    print()
                try:
                    if z[pixel+1]>0 and ((pixel+(1) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel+(1))
                        #print('+1')
                except:
                    print()
                try:
                    if z[pixel-1]>0 and ((pixel-(1) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-1)
                        #print('-1')
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
                if ob.solve(clusterxy)==True:
                    sigma=st.stdev(modeclusterenergy)
                    emean=np.mean(modeclusterenergy)
                    coeffvariation.append(sigma)
                    listofclustersize.append(n)
                    int1=((np.array(avenergy))/5)
                    int2=np.around(int1,0)
                    int3=list((int2)*5)
                    plotmodeaveenergy.append(st.mode(int3))
                    meanenergyincluster.append(np.mean(modeclusterenergy))

                    here.append(modeclusterenergy)

                print(ob.solve(clusterxy))
                if ob.solve(clusterxy)==False:
                    straight.append(0)
                if ob.solve(clusterxy)==True:
                    straight.append(1)
            if n==1:
                listofclustersize.append(n)
                int1=((np.array(avenergy))/5)
                int2=np.around(int1,0)
                int3=list((int2)*5)
                plotmodeaveenergy.append(st.mode(int3))


                #print('n:',n)
                #print(xytf)
                #print('n: ',n)
            #print('CLUSTER ENERGY:',clusterenergy)
        #if n>19 and n<21:
            #print('HEREEEEEEEEEEEEE')
            #print(roundformode)
            #plt.plot(roundformode,'x-')
            
            #print(x[g],y[g])
        g=g+1
        avenergy=[]
    '''meanavenergy = sum(avenergy)/len(avenergy)
    modeavenergy=st.mode(avenergy)
    maxavenergy=max(avenergy)
    minavenergy=min(avenergy)
    print("mean energy deposited of {}: {}".format(particle1,meanavenergy))
    print("mode energy deposited of {}: {}".format(particle1,modeavenergy))
    print("max energy deposited of {}: {}".format(particle1,maxavenergy))
    print("min energy deposited of {}: {}".format(particle1,minavenergy))'''





particle1 ="muon"
f=0
g=0
q=[]
avenergy=[]
total=0

if particle1=='muon':
    # opening and creating new .txt file 
    with open( 
        "/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV/muon_production_42_plot.dat", 'r') as r, open( 
            '/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV/muon_production_42_plot2.txt', 'w') as o: 
        
        for line in r: 
            #strip() function 
            if line.strip(): 
                o.write(line)                                                  #STRIPPING OUT THE BLANK LINES IN THE ORIGNAL AND CREATING A NEW ONE
                                                                            #BLANK LINES WERE BEING READ AND ADDED AS DATA.....
    # opening and creating new .txt file 
    with open( 
        "/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV/muon_production_39_plot.dat", 'r') as r, open( 
            '/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV/muon_production_39_plot2.txt', 'w') as o: 
        
        for line in r: 
            #strip() function 
            if line.strip(): 
                o.write(line) 
    



    y, x, z, err = np.loadtxt('/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV/muon_production_42_plot2.txt', unpack=True)
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
    y, x, z, err = np.loadtxt('/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV/muon_production_39_plot2.txt', unpack=True)


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


meanaveenergyincluster=[]
modeaveenergyincluster=[]
maxenergyincluster=[]
count=0

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
            clusterxy=[]
            while len(pixelist)>0:
                #print('here')
                if ((pixelist[0] in prevpixel)==True):
                    pixelist.remove(pixelist[0])
                    continue
                
                pixel=pixelist[0]
                clusterpixels.append(pixel)
                n=n+1

                clusterxy.append([x[pixel],y[pixel]])
                clusterenergy.append(z[pixel])
                modeclusterenergy.append(z[pixel]*primaries*0.00546875*0.00546875*0.1*(10**6))
                if jp==0:
                    print('heeeeeeeeeere:',z[pixel])
                #if g==10679:
                    #print(pixelist)
                    #print('HERE FOR EACH PIXEL')
                    #print(g)
                    #print(pixel,x[pixel],y[pixel])
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
                        #print('+257')
                except:
                    print()
                
                try:
                    if z[pixel+(256)]>0 and ((pixel+(256) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel+(256))
                        #print('+256')
                except:
                    print()
                try:
                    if z[pixel+(258)]>0 and ((pixel+(258) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel+(258))
                        #print('+258')
                        #print(z[pixel+(258)])
                except:
                    print()
                try:
                    if z[pixel-(257)]>0 and ((pixel-(257) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-(257))
                        #print('-257')
                except:
                    print()
                try:
                    if z[pixel-(256)]>0 and ((pixel-(256) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-(256))
                        #print('-256')
                except:
                    print()
                try:
                    if z[pixel-(258)]>0 and ((pixel-(258) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-(258))
                        #print('-258')
                except:
                    print()
                try:
                    if z[pixel+1]>0 and ((pixel+(1) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel+(1))
                        #print('+1')
                except:
                    print()
                try:
                    if z[pixel-1]>0 and ((pixel-(1) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-1)
                        #print('-1')
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
                if ob.solve(clusterxy)==True:
                    sigma=st.stdev(modeclusterenergy)
                    emean=np.mean(modeclusterenergy)
                    coeffvariation.append(sigma)
                    listofclustersize.append(n)
                    int1=((np.array(avenergy))/5)
                    int2=np.around(int1,0)
                    int3=list((int2)*5)
                    plotmodeaveenergy.append(st.mode(int3))
                    meanenergyincluster.append(np.mean(modeclusterenergy))

                    here.append(modeclusterenergy)

                print(ob.solve(clusterxy))
                if ob.solve(clusterxy)==False:
                    straight.append(0)
                if ob.solve(clusterxy)==True:
                    straight.append(1)
            if n==1:
                listofclustersize.append(n)
                int1=((np.array(avenergy))/5)
                int2=np.around(int1,0)
                int3=list((int2)*5)
                plotmodeaveenergy.append(st.mode(int3))


                #print('n:',n)
                #print(xytf)
                #print('n: ',n)
            #print('CLUSTER ENERGY:',clusterenergy)
        #if n>19 and n<21:
            #print('HEREEEEEEEEEEEEE')
            #print(roundformode)
            #plt.plot(roundformode,'x-')
            
            #print(x[g],y[g])
        g=g+1
        avenergy=[]
    '''meanavenergy = sum(avenergy)/len(avenergy)
    modeavenergy=st.mode(avenergy)
    maxavenergy=max(avenergy)
    minavenergy=min(avenergy)
    print("mean energy deposited of {}: {}".format(particle1,meanavenergy))
    print("mode energy deposited of {}: {}".format(particle1,modeavenergy))
    print("max energy deposited of {}: {}".format(particle1,maxavenergy))
    print("min energy deposited of {}: {}".format(particle1,minavenergy))'''


particle1 ="muon"
f=0
g=0
q=[]
avenergy=[]
total=0

if particle1=='muon':
    # opening and creating new .txt file 
    with open( 
        "/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV copy/muon_production_42_plot.dat", 'r') as r, open( 
            '/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV copy/muon_production_42_plot2.txt', 'w') as o: 
        
        for line in r: 
            #strip() function 
            if line.strip(): 
                o.write(line)                                                  #STRIPPING OUT THE BLANK LINES IN THE ORIGNAL AND CREATING A NEW ONE
                                                                            #BLANK LINES WERE BEING READ AND ADDED AS DATA.....
    # opening and creating new .txt file 
    with open( 
        "/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV copy/muon_production_39_plot.dat", 'r') as r, open( 
            '/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV copy/muon_production_39_plot2.txt', 'w') as o: 
        
        for line in r: 
            #strip() function 
            if line.strip(): 
                o.write(line) 
    



    y, x, z, err = np.loadtxt('/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV copy/muon_production_42_plot2.txt', unpack=True)
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
    y, x, z, err = np.loadtxt('/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV copy/muon_production_39_plot2.txt', unpack=True)


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


meanaveenergyincluster=[]
modeaveenergyincluster=[]
maxenergyincluster=[]
count=0

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
            clusterxy=[]
            while len(pixelist)>0:
                #print('here')
                if ((pixelist[0] in prevpixel)==True):
                    pixelist.remove(pixelist[0])
                    continue
                
                pixel=pixelist[0]
                clusterpixels.append(pixel)
                n=n+1

                clusterxy.append([x[pixel],y[pixel]])
                clusterenergy.append(z[pixel])
                modeclusterenergy.append(z[pixel]*primaries*0.00546875*0.00546875*0.1*(10**6))
                if jp==0:
                    print('heeeeeeeeeere:',z[pixel])
                #if g==10679:
                    #print(pixelist)
                    #print('HERE FOR EACH PIXEL')
                    #print(g)
                    #print(pixel,x[pixel],y[pixel])
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
                        #print('+257')
                except:
                    print()
                
                try:
                    if z[pixel+(256)]>0 and ((pixel+(256) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel+(256))
                        #print('+256')
                except:
                    print()
                try:
                    if z[pixel+(258)]>0 and ((pixel+(258) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel+(258))
                        #print('+258')
                        #print(z[pixel+(258)])
                except:
                    print()
                try:
                    if z[pixel-(257)]>0 and ((pixel-(257) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-(257))
                        #print('-257')
                except:
                    print()
                try:
                    if z[pixel-(256)]>0 and ((pixel-(256) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-(256))
                        #print('-256')
                except:
                    print()
                try:
                    if z[pixel-(258)]>0 and ((pixel-(258) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-(258))
                        #print('-258')
                except:
                    print()
                try:
                    if z[pixel+1]>0 and ((pixel+(1) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel+(1))
                        #print('+1')
                except:
                    print()
                try:
                    if z[pixel-1]>0 and ((pixel-(1) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-1)
                        #print('-1')
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
                if ob.solve(clusterxy)==True:
                    sigma=st.stdev(modeclusterenergy)
                    emean=np.mean(modeclusterenergy)
                    coeffvariation.append(sigma)
                    listofclustersize.append(n)
                    int1=((np.array(avenergy))/5)
                    int2=np.around(int1,0)
                    int3=list((int2)*5)
                    plotmodeaveenergy.append(st.mode(int3))
                    meanenergyincluster.append(np.mean(modeclusterenergy))
                    here.append(modeclusterenergy)

                print(ob.solve(clusterxy))
                if ob.solve(clusterxy)==False:
                    straight.append(0)
                if ob.solve(clusterxy)==True:
                    straight.append(1)
            if n==1:
                listofclustersize.append(n)
                int1=((np.array(avenergy))/5)
                int2=np.around(int1,0)
                int3=list((int2)*5)
                plotmodeaveenergy.append(st.mode(int3))


                #print('n:',n)
                #print(xytf)
                #print('n: ',n)
            #print('CLUSTER ENERGY:',clusterenergy)
        #if n>19 and n<21:
            #print('HEREEEEEEEEEEEEE')
            #print(roundformode)
            #plt.plot(roundformode,'x-')
            
            #print(x[g],y[g])
        g=g+1
        avenergy=[]
    '''meanavenergy = sum(avenergy)/len(avenergy)
    modeavenergy=st.mode(avenergy)
    maxavenergy=max(avenergy)
    minavenergy=min(avenergy)
    print("mean energy deposited of {}: {}".format(particle1,meanavenergy))
    print("mode energy deposited of {}: {}".format(particle1,modeavenergy))
    print("max energy deposited of {}: {}".format(particle1,maxavenergy))
    print("min energy deposited of {}: {}".format(particle1,minavenergy))'''


particle1 ="muon"
f=0
g=0
q=[]
avenergy=[]
herepixel=[]
total=0

if particle1=='muon':
    # opening and creating new .txt file 
    with open( 
        "/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV copy 2/muon_production_42_plot.dat", 'r') as r, open( 
            '/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV copy 2/muon_production_42_plot2.txt', 'w') as o: 
        
        for line in r: 
            #strip() function 
            if line.strip(): 
                o.write(line)                                                  #STRIPPING OUT THE BLANK LINES IN THE ORIGNAL AND CREATING A NEW ONE
                                                                            #BLANK LINES WERE BEING READ AND ADDED AS DATA.....
    # opening and creating new .txt file 
    with open( 
        "/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV copy 2/muon_production_39_plot.dat", 'r') as r, open( 
            '/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV copy 2/muon_production_39_plot2.txt', 'w') as o: 
        
        for line in r: 
            #strip() function 
            if line.strip(): 
                o.write(line) 
    



    y, x, z, err = np.loadtxt('/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV copy 2/muon_production_42_plot2.txt', unpack=True)
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
    y, x, z, err = np.loadtxt('/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV copy 2/muon_production_39_plot2.txt', unpack=True)


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

meanaveenergyincluster=[]
modeaveenergyincluster=[]
maxenergyincluster=[]
count=0

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
            modeclusterenergypixel=[]
            clusterxy=[]
            while len(pixelist)>0:
                #print('here')
                if ((pixelist[0] in prevpixel)==True):
                    pixelist.remove(pixelist[0])
                    continue
                
                pixel=pixelist[0]
                clusterpixels.append(pixel)
                n=n+1

                clusterxy.append([x[pixel],y[pixel]])
                clusterenergy.append(z[pixel])
                modeclusterenergy.append(z[pixel]*primaries*0.00546875*0.00546875*0.1*(10**6))
                modeclusterenergypixel.append(pixel)
                if jp==0:
                    print('heeeeeeeeeere:',z[pixel])
                #if g==10679:
                    #print(pixelist)
                    #print('HERE FOR EACH PIXEL')
                    #print(g)
                    #print(pixel,x[pixel],y[pixel])
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
                        #print('+257')
                except:
                    print()
                
                try:
                    if z[pixel+(256)]>0 and ((pixel+(256) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel+(256))
                        #print('+256')
                except:
                    print()
                try:
                    if z[pixel+(258)]>0 and ((pixel+(258) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel+(258))
                        #print('+258')
                        #print(z[pixel+(258)])
                except:
                    print()
                try:
                    if z[pixel-(257)]>0 and ((pixel-(257) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-(257))
                        #print('-257')
                except:
                    print()
                try:
                    if z[pixel-(256)]>0 and ((pixel-(256) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-(256))
                        #print('-256')
                except:
                    print()
                try:
                    if z[pixel-(258)]>0 and ((pixel-(258) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-(258))
                        #print('-258')
                except:
                    print()
                try:
                    if z[pixel+1]>0 and ((pixel+(1) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel+(1))
                        #print('+1')
                except:
                    print()
                try:
                    if z[pixel-1]>0 and ((pixel-(1) in prevpixel)==False):
                        check=1
                        pixelist.append(pixel-1)
                        #print('-1')
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
                if ob.solve(clusterxy)==True:
                    sigma=st.stdev(modeclusterenergy)
                    emean=np.mean(modeclusterenergy)
                    coeffvariation.append(sigma)
                    listofclustersize.append(n)
                    int1=((np.array(avenergy))/5)
                    int2=np.around(int1,0)
                    int3=list((int2)*5)
                    plotmodeaveenergy.append(st.mode(int3))
                    meanenergyincluster.append(np.mean(modeclusterenergy))




                    here.append(modeclusterenergy)
                    herepixel.append(modeclusterenergypixel)

                print(ob.solve(clusterxy))
                if ob.solve(clusterxy)==False:
                    straight.append(0)
                if ob.solve(clusterxy)==True:
                    straight.append(1)
            if n==1:
                listofclustersize.append(n)
                int1=((np.array(avenergy))/5)
                int2=np.around(int1,0)
                int3=list((int2)*5)
                plotmodeaveenergy.append(st.mode(int3))


                #print('n:',n)
                #print(xytf)
                #print('n: ',n)
            #print('CLUSTER ENERGY:',clusterenergy)
        #if n>19 and n<21:
            #print('HEREEEEEEEEEEEEE')
            #print(roundformode)
            #plt.plot(roundformode,'x-')
            
            #print(x[g],y[g])
        g=g+1
        avenergy=[]
    '''meanavenergy = sum(avenergy)/len(avenergy)
    modeavenergy=st.mode(avenergy)
    maxavenergy=max(avenergy)
    minavenergy=min(avenergy)
    print("mean energy deposited of {}: {}".format(particle1,meanavenergy))
    print("mode energy deposited of {}: {}".format(particle1,modeavenergy))
    print("max energy deposited of {}: {}".format(particle1,maxavenergy))
    print("min energy deposited of {}: {}".format(particle1,minavenergy))'''






weights=[]

for i in range(0,len(totalenergyincluster)):

    weights.append(1/len(totalenergyincluster))  
binwidth=50
fig, ax=plt.subplots()
ax = plt.gca()
ax.set_xlim([0,2000])
plt.xlabel("Total Energy in Cluster - KeV")
plt.ylabel('Probability')
#ax.hist(totalenergyincluster, bins=150, weights=weights)
ax.hist(totalenergyincluster, bins=range(int(min(totalenergyincluster)), int(max(totalenergyincluster) + binwidth), binwidth),weights=weights)
plt.savefig('/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV/total_energy_in_cluster_hist.png',bbox_inches='tight', dpi=1000)



weights=[]

for i in range(0,len(coeffvariation)):

    weights.append(1/len(coeffvariation))
binwidth=5
fig, ax=plt.subplots()
ax = plt.gca()
ax.set_xlim([0,130])
plt.xlabel("Standard Deviation - KeV")
plt.ylabel('%N')

#ax.hist(totalenergyincluster, bins=150, weights=weights)
#ax.hist(coeffvariation, bins=(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5), color='b')
#counts, bins, bars = ax.hist(coeffvariation, bins=(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5), color='b')
ax.hist(coeffvariation, bins=range(int(min(coeffvariation)), int(max(coeffvariation) + binwidth), binwidth), color='b')
counts, bins, bars =ax.hist(coeffvariation, bins=range(int(min(coeffvariation)), int(max(coeffvariation) + binwidth), binwidth), color='b')
plt.savefig('/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV/{}_coeff_variation_in_cluster_hist.png'.format(particle1),bbox_inches='tight', dpi=1000)
#print(counts, bins, bars)

#coeffsave=np.column_stack((counts, bins))
coeffsave=[[],[]]
for q in range(0,len(counts)):
    coeffsave[0].append(counts[q])
    coeffsave[1].append(bins[q])

with open("/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV/muoncoeffsave.csv", "w") as txt_file:
    for q in range(0,len(counts)):
        txt_file.write("%g\t%g\n"%(float(coeffsave[1][q]),float(coeffsave[0][q]))) # works with any number of elements in a line



#######50MeV
for j in range(0,55,5):
    prob=0
    for i in range(0,len(counts)):
        if bins[i]>=j:
            prob+=counts[i]
    print('Probs of muon of Std Dev >= %gKeV'%(j), prob)



print((bins))
'''with open( 
    "/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV/muoncoeffsave.csv", 'w') as muoncoeffsave:
    savetxt("/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV/muoncoeffsave.csv", coeffsave, delimiter=',')'''


weights=[]

for i in range(0,len(plotmodeaveenergy)):

    weights.append(1/len(plotmodeaveenergy))  
binwidth=5
fig, ax=plt.subplots()
ax = plt.gca()
ax.set_xlim([0,170])
plt.xlabel("Mode Energy in Cluster - KeV")
plt.ylabel('%N')

#ax.hist(totalenergyincluster, bins=150, weights=weights)
ax.hist(plotmodeaveenergy, bins=range(int(min(plotmodeaveenergy)), int(max(plotmodeaveenergy) + binwidth), binwidth), color='b')
counts, bins, bars = ax.hist(plotmodeaveenergy, bins=range(int(min(plotmodeaveenergy)), int(max(plotmodeaveenergy) + binwidth), binwidth), color='b')
plt.savefig('/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV/{}_mode_energy_in_cluster_hist.png'.format(particle1),bbox_inches='tight', dpi=1000)
print(counts, bins, bars)

modesave=[[],[]]
for q in range(0,len(counts)):
    modesave[0].append(counts[q])
    modesave[1].append(bins[q])

#print(tracksave)
with open("/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV/muonmodesave.csv", "w") as txt_file:
    for q in range(0,len(counts)):
        txt_file.write("%g\t%g\n"%(float(modesave[1][q]),float(modesave[0][q]))) # works with any number of elements in a line



weights=[]

for i in range(0,len(listofclustersize)):

    weights.append(1/len(listofclustersize))  
fig, ax=plt.subplots()   
binwidth=1
plt.hist(listofclustersize, bins=range(int(min(listofclustersize)), int(max(listofclustersize) + binwidth)), width=1,color='b')
counts, bins, bars = plt.hist(listofclustersize, bins=range(int(min(listofclustersize)), int(max(listofclustersize) + binwidth)), width=1,color='b')
plt.xlabel("Size of Cluster - pixels")
plt.ylabel('%N')
xlim=40
ax.set_xlim([0,xlim])   
plt.xticks(np.arange(0, xlim, step=5))
                                                     #number of cluster sizes
#ax.set_ylim([0,20])
plt.savefig('/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV/{}_clustersizes.png'.format(particle1), bbox_inches='tight')


tracksave=[[],[]]
for q in range(0,len(counts)):
    tracksave[0].append(counts[q])
    tracksave[1].append(bins[q])

#print(tracksave)
with open("/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV/muontracksave.csv", "w") as txt_file:
    for q in range(0,len(counts)):
        txt_file.write("%g\t%g\n"%(float(tracksave[1][q]),float(tracksave[0][q]))) # works with any number of elements in a line




for j in range(1,16,1):
    prob=0
    for i in range(0,len(counts)):
        if bins[i]>=j:
            prob+=counts[i]
    print('Probs of muon of size >= %g pixels'%(j), prob)



#here2=np.array(here)
#here3=np.around(here2,0)
#print('HERE:',here3)

print(xy)
fig, ax=plt.subplots() 
ax.hist(straight)

'''fig, ax=plt.subplots()  
plt.hist2d(x, y,(257,257), weights=z, norm=colors.LogNorm())
plt.xlabel("cm")
plt.ylabel('cm')
cb = plt.colorbar()
tick_locator = ticker.MaxNLocator(nbins=10)
cb.locator = tick_locator
cb.update_ticks()
#plt.savefig('{}_usrbin.png'.format(particle1), bbox_inches='tight')
#plt.show()
plt.colorbar()
plt.savefig('{}_log_usrbin.png'.format(particle1), bbox_inches='tight')'''





weights=[]

for i in range(0,len(meanenergyincluster)):

    weights.append(1/len(meanenergyincluster)) 
binwidth=5
fig, ax=plt.subplots()   
ax = plt.gca()
ax.hist(meanenergyincluster, bins=range(int(min(meanenergyincluster)), int(max(meanenergyincluster) + binwidth),binwidth),color='b')
counts, bins, bars = ax.hist(meanenergyincluster, bins=range(int(min(meanenergyincluster)), int(max(meanenergyincluster) + binwidth),binwidth),color='b')
plt.xlabel("Mean energy - KeV")
plt.ylabel('%N')
xlim=250
ax.set_xlim([0,xlim])   
                                                     #number of cluster sizes
#ax.set_ylim([0,20])
plt.savefig('/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV/{}_meanenergy.png'.format(particle1), bbox_inches='tight', dpi=1000)

meansave=[[],[]]
for q in range(0,len(counts)):
    meansave[0].append(counts[q])
    meansave[1].append(bins[q])

#print(tracksave)
with open("/Users/lukecalvin/2023/eli_np_muon_primaries_1.0GeV/muonmeansave.csv", "w") as txt_file:
    for q in range(0,len(counts)):
        txt_file.write("%g\t%g\n"%(float(meansave[1][q]),float(meansave[0][q]))) # works with any number of elements in a line


#50MeV
for j in range(0,55,5):
    prob=0
    for i in range(0,len(counts)):
        if bins[i]>=j:
            prob+=counts[i]
    print('Probs of muon of energy >= %gKeV'%(j), prob)



print(len(listofclustersize))





slope=[]
b1full=[]
for i in range(0,len(here)):
    maxx=max(here[i])

    for j in range(0,len(here[i])):
        
        here[i][j]=here[i][j]/maxx
    #formean=here[i][:]
    #formean.remove(1.0)
    tim = np.arange(0,len(here[i]))
    m1, b1 = np.polyfit(tim, here[i], 1)
    slope.append(abs(m1))
    b1full.append(b1)
    stand=st.stdev(here[i])
    #if (1-(np.mean(formean)))<0.2:
    #   print(i+1) 
    #print('Track Number:',i+1,'\nSlope:',m1)
    normstand=stand/(len(here[i]))        #standard deviation normalised into standard deviation of the slope. this is due to the stdev being from the y axis. m=rise/run, so stdev needs divid by run
    if abs(slope[i])<normstand:
        print('slope:',slope[i])
        print('stand:',stand)
        print('run:',len(here[i]))
        print('normstand:',normstand)
        print(i+1)



fig, ax=plt.subplots()
ax = plt.gca()  
plt.ylim(0,1.2)
plt.xlim(0,10)
print(len(here))
for i in range(0,10):

    #ax.plot(here[i],'o')
    ax.plot(here[i],'o-',c='C%g'%(i),label='%g'%(i+1))
    ax.plot(slope[i]*tim+b1full[i],ls='dotted',c='C%g'%(i),lw=2,label='%g slope=%1.3f'%(i+1,slope[i]))
    plt.legend()

print(here)
print(herepixel)
diffs=[]
diff=[]
for i in range(0,len(here)):


    for j in range(0,len(here[i])-1):
        diff.append(abs(here[i][j+1]-here[i][j]))
    meandiff=np.mean(diff)
    diffs.append(meandiff)
meandiffs=np.mean(diffs)

print('mean slope:', np.mean(slope))
print('max slope: ', max(slope))
print('min slope: ', min(slope))
plt.show()
