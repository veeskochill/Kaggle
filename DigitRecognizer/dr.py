import math
import numpy as np
import numpy.linalg as LA
import numpy.fft as FFT

class pixel(object):
	def __init__(self,color,x,y):
		self.x = x
		self.y = y
		self.color = color

class digit(object):
	def __init__(self, value, param):
		self.value = value
		self.param = param

def regions(data):
	points = []
	sizes = [0 for z in range(10)]
	xaverages = [0 for z in range(10)]
	yaverages = [0 for z in range(10)]

	fptr = open("dplot.dat","r")
	for z in range(100):
		dfile = fptr.readline()
		words = dfile.split()
		#print ("%s" % words[0])
		actual = int(words[0])
		x = float(words[1])
		y = float(words[2])
		sizes[actual] += 1
		xaverages[actual] += x
		yaverages[actual] += y

		points.append(pixel(actual, x, y))
	fptr.close()
	#analyze data
	for z in range(10):
		if sizes[z] >0:
			xaverages[z] /= sizes[z] 
			yaverages[z] /= sizes[z]
	radii = [0 for z in range(10)]
	for z in range(100):
		local = (points[z].x - xaverages[points[z].color])**2 + (points[z].y - yaverages[points[z].color])**2
		if radii[points[z].color] < local:
			radii[points[z].color] = local
	fptr2 = open("regions.dat","w")
	for z in range(10):
		fptr2.write("%d %f %f %f\n" % (z, xaverages[z], yaverages[z], radii[z]**(0.5)))
	fptr2.close()


def Fourier(actual, data):
	fdata = []
	for item in data:
		fdata.append(item.color)
	myft = FFT.fft(fdata)
	fptr = open("fdata1.dat","a")
	for index, item in enumerate(myft):
		fptr.write("%d %f\n" % (index%28, item.real))
	fptr.close()



def covMatrix(actual, data):
	x_bar = 0 #mean x value
	y_bar = 0 #mean y value
	w_sum = 0 #sum of weights
	for item in data:
		x_bar += item.x*item.color
		y_bar += item.y*item.color
		w_sum += item.color
	x_bar /= w_sum #len(data)
	y_bar /= w_sum #len(data)

	vc = [[0 for x in range(2)] for x in range(2)]
	vc_xy =0
	vc_xx = 0
	vc_yy =0
	for item in data:
		vc_xy += (item.x - x_bar)*(item.y - y_bar)*item.color
		vc_xx += (item.x - x_bar)*(item.x - x_bar)*item.color
		vc_yy += (item.y - y_bar)*(item.y - y_bar)*item.color

	n = 28*28
	vc[0][0] = (vc_xx/(w_sum-1))
	vc[0][1] = (vc_xy/(w_sum-1))
	vc[1][0] = (vc_xy/(w_sum-1))
	vc[1][1] = (vc_yy/(1-w_sum))

	eigs = LA.eigvals(vc)
	#eigs /= LA.norm(eigs)
	print ("%d : %f, %f" % (actual, eigs[0], eigs[1]))
	fptr = open("dplot.dat","a")
	fptr.write("%d %f %f\n" % (actual, eigs[0], eigs[1]))
	fptr.close()
	return eigs

#
def linCorr(actual, data):
	#Calculate and store feature vector
	#Calculate linear correlation coefficient
	#n(Sum(xy) - Sum(x)Sum(y) / 
	#	sqrt(nSum(x^2)-Sum(x)^2)sqrt(nSum(y^2)-Sum(y)^2))

	sumxy = 0
	sumx = 0
	sumy = 0
	sumx2 = 0
	sumy2 = 0
	for item in data:
		sumx += item.x
		sumy += item.y
		sumxy += item.x*item.y
		sumx2 += item.x*item.x
		sumy2 += item.y*item.y

	#print ("Sum X = %d" % sumx)
	#print ("Sum X^2 = %d" % sumx2)
	#print ("Sum y = %d" % sumy)
	#print ("Sum y^2 = %d" % sumy2)
	#print ("Sum xy = %d" % sumxy)

	n = 28*28

	#print ("numerator %f" % (n*sumxy - sumx*sumy))
	#print ("denominator %f" % (math.sqrt(n*sumx2 - sumx*sumx)*math.sqrt(n*sumy2 - sumy*sumy)))

	coeff = (n*sumxy - sumx*sumy)/(math.sqrt(n*sumx2 - sumx*sumx)*math.sqrt(n*sumy2 - sumy*sumy))
	print ("%d : %f" % (actual, coeff))
	return coeff


vectors = []

cutoff = 200 #minimum grayscale value

f = open("train.csv","r")


# remove first label's line
img = f.readline()

for zeta in range(0,500):
	img = f.readline()

	data = []
	fdata = []
	actual =0
	#Read data and store
	imgs = img.split(',')

	for index, item in enumerate(imgs):
		if index ==0 :
			actual = int(item)

		else:
			fx =  index//28
			fy = index%28
			fp = pixel(int(item), fx,fy)
			fdata.append(fp)
			if int(item) > cutoff:
				x =  index//28
				y = index%28
				p = pixel(int(item), x,y)
				data.append(p)
				#print("%d %d : %d " % (x, y, int(item)))
	Fourier(actual,fdata)
	covMatrix(actual, data)
#regions(data)
#	vectors.append(linCorr(actual, data)

f.close()
