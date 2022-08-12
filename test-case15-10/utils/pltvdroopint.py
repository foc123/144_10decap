#-- plot voltage droop integral
#-- NOTE: only python2 can run this code

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


if len(sys.argv) != 2:
    print("Plot voltage droop integral map")
    print("Usage: python pltvdroopint.py <voltage_droop_integral_map_csv_file>")
    sys.exit(1)


#-- read in csv format voltage droop integral file 
#-- format: xloc\tyloc\tvoltage
def ReadCSVfile(filename):
    array = np.genfromtxt(filename)
    return array

#-- read in csv format voltage droop integral file 
#-- return min, avg, max, and total vdi
def getVDIinfo(filename):
    res_array = np.genfromtxt(filename)
    vdi_array = res_array[:,2]
    return np.min(vdi_array), np.average(vdi_array), np.max(vdi_array), np.sum(vdi_array)

res_array = ReadCSVfile(sys.argv[1])
npts = len(res_array)
x = res_array[:,0]
y = res_array[:,1]
z = res_array[:,2]

nv = 0
i = 0
no = len(z)
while i < no:
    if z[i] > 0.0:
        nv += 1
    i += 1
print("No. of Node with voltage droop violations: %d/%d [%.2f%%]" % (nv,no, 100*float(nv)/float(no)))

total_vdi = np.sum(z)
min_vdi = np.min(z)
avg_vdi = np.average(z)
max_vdi = np.max(z)
print("Min voltage droop integral: %g[V*S]" % min_vdi)
print("Avg voltage droop integral: %g[V*S]" % avg_vdi)
print("Max voltage droop integral: %g[V*S]" % max_vdi)
print("Total voltage droop integral: %g[V*S]" % total_vdi)

fig, (ax2) = plt.subplots(nrows=1)

# ----------
# Tricontour
# ----------
# Directly supply the unordered, irregularly spaced coordinates
# to tricontour.

ax2.tricontour(x, y, z, levels=14, linewidths=0.5, colors='k')
cntr2 = ax2.tricontourf(x, y, z, levels=14, cmap="RdBu_r")

fig.colorbar(cntr2, ax=ax2)
ax2.plot(x, y, 'ko', ms=3)

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)


plt.subplots_adjust(hspace=0.5)
#plt.show()


#-- plot histogram of node vdis
print("No. of VDIs: %d/%d" % (nv,no))

n_bins = 20

# Creating histogram
fig, axs = plt.subplots(1, 1)

axs.hist(z, bins = n_bins)

# Show plot
plt.show()

