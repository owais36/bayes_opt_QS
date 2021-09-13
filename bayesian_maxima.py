import sys
import numpy as np
import csv
import matplotlib.pyplot as plt
from numpy.random import normal
from numpy.random import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from sklearn.gaussian_process.kernels import RBF,ConstantKernel,WhiteKernel,ExpSineSquared
import warnings
warnings.filterwarnings('error')


def convert_file2_matrix(raster_file):
    timestamp = []
    lat = []
    lon = []
    alt= []
    flag = []
    P_el = []
    P_az = []
    signal = []

    file_name = raster_file
    n=0

    csv_file = csv.reader(open(file_name,'r'),delimiter=',')
    for row in csv_file:
        if row[0][0] != '%':
            timestamp.append(row[0])
            lat.append(row[1])
            lon.append(row[2])
            alt.append(row[3])
            flag.append(row[4])
            P_az.append(row[5])
            P_el.append(row[6])
            signal.append(row[7])
            

    del flag[0]
    del signal[0]
    del P_az[0]
    del P_el[0]
    signal2 = []
    az2 = []
    el2 = []
    for i in range(len(flag)):
        signal[i]=float(signal[i])
        P_az[i]=float(P_az[i])
        P_el[i]=float(P_el[i])



    z_val = list(zip(flag,signal,P_az,P_el))
    val = []
    val_az = []
    val_el = []
    count = 0
    j = z_val[0][0]

    for i in z_val:
        if i[0] == j:
            val.append(i[1])
            val_az.append(i[2])
            val_el.append(i[3])
        else:
            signal2.append(list(val))
            az2.append(list(val_az))
            el2.append(list(val_el))
            val[:]=[]
            val_az[:]=[]
            val_el[:]=[]
            j = i[0]
            val.append(i[1])
            val_az.append(i[2])
            val_el.append(i[3])


    signal2.append(list(val))
    az2.append(list(val_az))
    el2.append(list(val_el))


    for i in range(len(az2)):
        if i == 0:
            continue
        if i%2 == 0:
            az2[i] = az2[i][::-1]
            signal2[i] = signal2[i][::-1]
            el2[i] = el2[i][::-1]

    for x in range(len(signal2)):
        del signal2[x][1300:]
    for x in range(len(az2)):
        del az2[x][1300:]
    for x in range(len(el2)):
        del el2[x][1300:]

    signal2 = np.array(signal2,dtype=float)
    az2=np.array(az2,dtype=float)
    el2=np.array(el2,dtype=float)

    area = signal2.shape

    rt_data = np.zeros((3,area[0],area[1]))
    rt_data[0,:,:] = az2
    rt_data[1,:,:] = el2
    rt_data[2,:,:] = signal2

    return rt_data

def find_ranges(matrix):
        min_val = np.min(matrix[0])
        max_val = np.max(matrix[0])
        for x in range(len(matrix)):
            minima = np.min(matrix[x])
            maxima = np.max(matrix[x])
            if min_val < minima:
                min_val = minima
            if max_val < maxima:
                max_val = maxima
        
        return min_val,max_val

class bayesian_search:
    def __init__(self,rt_data):
        self.rt_data = rt_data
        self.signal = rt_data[2,:,:]
        self.azimuth = rt_data[0,:,:]
        self.elevation = rt_data[1,:,:]
        self.az_min,self.az_max = find_ranges(self.azimuth)
        self.el_min,self.el_max = find_ranges(self.elevation.transpose())
        self.fig = 0

        self.distance = 0
        self.ff_radius = 100
        self.budget = 150

        _,self.ground_truth = self.confirm_maxima()
        self.az_array = []
        self.az_error = []
        self.el_array = []
        self.el_error = []
        self.dist_array = []

    def reverse_lookup(self,az,el):
        curr_x = 0
        curr_y = 0

        min_diff = np.absolute(az - self.azimuth[0,0])
        for i in range(self.azimuth.shape[1]):
            diff = np.absolute(az - self.azimuth[0,i])
            if diff < min_diff:
                min_diff = diff
                curr_y = i

        min_diff = np.absolute(el - self.elevation[0,0])
        for i in range(self.elevation.shape[0]):
            diff = np.absolute(el - self.elevation[i,0])
            if diff < min_diff:
                min_diff = diff
                curr_x = i

        return (curr_x,curr_y)

    def surrogate(self,model,X):
        with catch_warnings():
            simplefilter("ignore")
            return model.predict(X,return_std=True)

    def acquistion(self,X,Xsamples,model):
        yhat,_ = self.surrogate(model,X)
        best = max(yhat)
        mu,std = self.surrogate(model,Xsamples)
        mu = mu[:,0]
        probs = norm.cdf((mu-best)/(std+1E-9))
        return probs

    def acquistionEI(self,X,Xsamples,model):
        yhat,_ = self.surrogate(model,X)
        best = max(yhat)
        mu,std = self.surrogate(model,Xsamples)
        mu = mu[:,0]
        Z = (mu-best)/(std+1E-9)
        probs = (mu-best)*norm.cdf(Z) + std*norm.pdf(Z)
        return probs

    def opt_acquisition(self,X,y,model):
        az = np.random.uniform(low=self.az_min,high=self.az_max,size=(10000,))
        el = np.random.uniform(low=self.el_min,high=self.el_max,size=(100,))
        Xsamples = []
        for i in range(len(az)):
            for j in range(len(el)):
                Xsamples.append([az[i],el[j]])
        scores = self.acquistionEI(X,Xsamples,model)
        #scores = self.acquistion(X,Xsamples,model)
        ix = np.argmax(scores)
        return Xsamples[ix]

    def opt_acquisition2(self,X,y,model,point):
        az_low = point[0]-1
        if az_low < self.az_min:
            az_low = self.az_min
        az_high = point[0]+1
        if az_high > self.az_max:
            az_high = self.az_max
        el_low = point[1]-1
        if el_low < self.el_min:
            el_low = self.el_min
        el_high = point[1]+1
        if el_high > self.el_max:
            el_high = self.el_max
        az = np.random.uniform(low=az_low,high=az_high,size=(100,))
        el = np.random.uniform(low=el_low,high=el_high,size=(100,))
        Xsamples = []
        for i in range(len(az)):
            for j in range(len(el)):
                Xsamples.append([az[i],el[j]])
        scores = self.acquistionEI(X,Xsamples,model)
        #scores = self.acquistion(X,Xsamples,model)
        ix = np.argmax(scores)
        return Xsamples[ix]

    def plot_model(self,X,y,model,pause=True,show_plot=False,name=""):
        #Plots whatever model we have
        theta = np.arange(az_min,az_max,0.1)
        phi = np.arange(el_min,el_max,0.1)
        Xsamples = np.zeros(((len(theta)*len(phi)),2))
        th_count = 0
        ph_count = 0
        for i in range(len(theta)*len(phi)):
            Xsamples[i,0] = theta[th_count]
            Xsamples[i,1] = phi[ph_count]
            th_count += 1
            if th_count == len(theta):
                th_count = 0
                ph_count += 1 #Need to check ph_count doesn't exceed phi index

        ysamples,_ = self.surrogate(model,Xsamples)
        ysamples = ysamples.reshape(len(phi),len(theta))

        level = np.linspace(-75,-11,120)
        cs = plt.contourf(theta,phi,ysamples,levels=level,extend = 'both')
        plt.colorbar()
        plt.savefig('./output/Bayes1/{}'.format(self.fig))
        if show_plot==True:
            plt.show(block=pause)
            plt.pause(1)
        plt.close()
        
    def generate_model(self,az,el):
        curr_x,curr_y = self.reverse_lookup(az,el)
        curr_az = self.azimuth[curr_x,curr_y]
        curr_el = self.elevation[curr_x,curr_y]
        curr_signal = self.signal[curr_x,curr_y]

        #Defining the first point for surrogate function
        X = np.array([[curr_az,curr_el]])
        y = np.array([[curr_signal]])
        
        #Gaussian process model
        ker = RBF(2.0,length_scale_bounds="fixed")
        
        self.model = GaussianProcessRegressor(kernel = ker)
        self.model.fit(X,y)

        #range point
        x=[curr_az,curr_el]
        old_x = (curr_x,curr_y)
        for i in range(1000):
            self.fig = i+1
            # opt_acquisition2 is restricted sampling whereas,
            # opt_acqusition i un-restricted sampling
            x = self.opt_acquisition2(X,y,self.model,x)
            #x = self.opt_acquisition(X,y,self.model)
            curr_x,curr_y = self.reverse_lookup(x[0],x[1])
            actual = self.signal[curr_x,curr_y]
            est,_ = self.surrogate(self.model,[x])
            print("Sampling point:{}".format(x))
            print("Posterior Value:{}".format((est)))
            print("Observed Value:{}".format(actual))
            print("------------------------")
            X = np.vstack((X,[x]))
            y = np.vstack((y,[actual]))
            self.model.fit(X,y)
            #self.plot_model(X,y,self.model,pause=False,show_plot=False)

            #Calculating and saving distance
            self.distance += self.get_distance(old_x,(curr_x,curr_y))
            if self.distance > self.budget:
                print("Distance limit reached!")
                break
            old_x = (curr_x,curr_y)

            ######################################
            ix = np.argmax(y)
            x_ = X[ix]
            self.az_array.append(x_[0])
            self.az_error.append(self.ground_truth[0]-x_[0])
            self.el_array.append(x_[1])
            self.el_error.append(self.ground_truth[1]-x_[1])
            self.dist_array.append(self.distance)

            ######################################
        
        self.plot_model(X,y,self.model,pause=True,show_plot=True)

        #For printing the best result
        ix = np.argmax(y)
        print("*********Result for Maxima*************")
        print('Maxima location:{}, signal value={}'.format(X[ix], y[ix]))
        print("****************************")
        return (X,y)

    def get_distance(self,p1,p2):
        az1 = np.deg2rad(self.rt_data[0,p1[0],p1[1]])
        el1 = np.deg2rad(self.rt_data[1,p1[0],p1[1]])
        az2 = np.deg2rad(self.rt_data[0,p2[0],p2[1]])
        el2 = np.deg2rad(self.rt_data[1,p2[0],p2[1]])
        try:
            dist = self.ff_radius * np.sqrt(2) * np.sqrt(1 - np.cos(az1)*np.cos(az2)*np.cos(el1-el2) - np.sin(az1)*np.sin(az2))
        except Warning:
            dist = 0
        return dist

    def confirm_maxima(self):
        signal = self.rt_data[2,:,:]
        val = signal[0,0]
        row,col = 0,0
        x,y = signal.shape[0],signal.shape[1]
        for i in range(x):
            for j in range(y):
                cur_val = signal[i,j]
                if cur_val > val:
                    val = cur_val
                    row = i
                    col = j

        return val,(self.azimuth[row,col],self.elevation[row,col])

raster_file = "./input/FID1110_07_raster.csv"
rt_data = convert_file2_matrix(raster_file)
signal = rt_data[2,:,:]
az = rt_data[0,:,:]
el = rt_data[1,:,:]
area = signal.shape


az_min,az_max = find_ranges(az)
el_min,el_max = find_ranges(el.transpose())

#Bayesian search part
start_az = (az_max-az_min)*0.2 + az_min
start_el = (el_max-el_min)*0.25 + el_min
bayes = bayesian_search(rt_data)
#Specify search budget here
bayes.budget = 150
X,y = bayes.generate_model(start_az,start_el)

# x_ = [p[0] for p in X]
# y_ = [p[1] for p in X]
# plt.plot(x_,y_,marker = '*')
# plt.title("Drone path (Random Search)")
# plt.xlabel("Azimuth (Degrees)")
# plt.ylabel("Elevation (Degrees)")
# plt.show()

print(len(bayes.az_array))
print(len(bayes.dist_array))
plt.clf()
plt.plot(bayes.dist_array,bayes.az_array,marker='*',label = "Main Beam (Azimuth)")
plt.plot(bayes.dist_array,bayes.az_error,marker='*',label = "Azimuth Error (from Ground Truth)")
plt.plot(bayes.dist_array,bayes.el_array,marker='*',label = "Main Beam (Elevation)")
plt.plot(bayes.dist_array,bayes.el_error,marker='*',label = "Elevation Error (from Ground Truth)")
plt.legend()
plt.title("BO")
plt.xlabel("meters")
plt.ylabel("degrees")
plt.show()
