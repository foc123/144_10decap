#-- plot voltage droop integral
#-- NOTE: only python2 can run this code

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


if len(sys.argv) != 2:
    print("print voltage droop integral result")
    print("Usage: python prvdi.py <voltage_droop_integral_map_csv_file>")
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

vdi_min, vdi_avg, vdi_max, vdi_total = getVDIinfo(sys.argv[1])
print("Min voltage droop integral: %g[V*S]" % vdi_min)
print("Avg voltage droop integral: %g[V*S]" % vdi_avg)
print("Max voltage droop integral: %g[V*S]" % vdi_max)
print("Total voltage droop integral: %g[V*S]" % vdi_total)
