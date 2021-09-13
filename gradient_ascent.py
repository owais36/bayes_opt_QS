import numpy as np 
import csv
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def convert_file2_matrix(raster_file,file_range):
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

    csv_file = csv.reader(open(file_name),delimiter=',')
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
        del signal2[x][file_range:]
    for x in range(len(az2)):
        del az2[x][file_range:]
    for x in range(len(el2)):
        del el2[x][file_range:]

    signal2 = np.array(signal2,dtype=float)
    az2=np.array(az2,dtype=float)
    el2=np.array(el2,dtype=float)

    area = signal2.shape

    rt_data = np.zeros((3,area[0],area[1]))
    rt_data[0,:,:] = az2
    rt_data[1,:,:] = el2
    rt_data[2,:,:] = signal2

    with open("output/signal.csv",'w') as f:
        wr = csv.writer(f)
        for i in range(len(signal2)):
            wr.writerow(signal2[i])

    with open("output/az.csv",'w') as f:
        wr = csv.writer(f)
        for i in range(len(az2)):
            wr.writerow(az2[i])

    with open("output/el.csv",'w') as f:
        wr = csv.writer(f)
        for i in range(len(el2)):
            wr.writerow(el2[i])

    return rt_data

class gradient_ascent():
    def __init__(self,rt_data):
        self.rt_data = rt_data
        self.x_max = rt_data[2,:,:].shape[0]
        self.x_min = 0
        self.y_max = rt_data[2,:,:].shape[1]
        self.y_min = 0
        self.alpha = 0.2 #Learning rate for gradient ascent
        self.signal = rt_data[2,:,:]
        self.azimuth = rt_data[0,:,:]
        self.elevation = rt_data[1,:,:]

        self.func_width_az = 319    
        self.func_width_el = 11

        self.az_min,self.az_max = self.find_ranges(self.azimuth) 
        self.el_min,self.el_max = self.find_ranges(self.elevation.transpose())

        self.distance = 0
        self.ff_radius = 100

        _,peak = self.confirm_maxima()
        self.ground_truth = (rt_data[0,peak[0],peak[1]],rt_data[1,peak[0],peak[1]])
        self.az_array = []
        self.az_error = []
        self.el_array = []
        self.el_error = []
        self.dist_array = []
        self.saved_path = []

    def power_law(self,x,a,c):
        return (a*np.power(x, 2) + c)
        

    def find_ranges(self,matrix):
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

    def find_gradient(self,x_val,y_val,theta):
        pars, cov = curve_fit(f=self.power_law, xdata=x_val, ydata=y_val, bounds=(-np.inf, np.inf))
        a = pars[0]
        # b = pars[1]
        # c = pars[2]

        #Calculating the derivative of function at location theta
        der = (2*a)*theta
        return der

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

    def find_maxima(self,start):
        curr_x = start[0]
        curr_y = start[1]
        curr_az = self.azimuth[start[0],start[1]]
        curr_el = self.elevation[start[0],start[1]]

        az_vals = []
        el_vals = []
        sig_vals_az = []
        sig_vals_el = []
        maxima = False

        itr = 0

        plot_signal = np.zeros(self.signal.shape)
        plot_signal[:] = self.signal

        while(not maxima):
            itr += 1
            if itr == 20:
                break
            print("Itr!")
            #For Azimuth axis
            x_range_min = curr_y - int(((self.func_width_az - 1)/2))
            x_range_max = curr_y + int(((self.func_width_az - 1)/2))
            if x_range_min < 0:
                x_range_min = 0
            if x_range_max > 1299:
                x_range_max = 1299

            az_vals = self.azimuth[curr_x,x_range_min:x_range_max]
            sig_vals_az = self.signal[curr_x,x_range_min:x_range_max]
            delta_az = self.find_gradient(az_vals,sig_vals_az,curr_az)
            
            for p in az_vals:
                self.saved_path.append((p,self.elevation[curr_x,x_range_min]))

            self.az_array.append(curr_az)
            self.az_error.append(self.ground_truth[0]-curr_az)

            #For Elevation axis
            y_range_min = curr_x - int(((self.func_width_el - 1)/2))
            y_range_max = curr_x + int(((self.func_width_el - 1)/2))
            if y_range_min < 0:
                y_range_min = 0
            if y_range_max > 45:
                y_range_max = 45
            
            el_vals = self.elevation[y_range_min:y_range_max,curr_y]
            sig_vals_el = self.signal[y_range_min:y_range_max,curr_y]
            delta_el = self.find_gradient(el_vals,sig_vals_el,curr_el)

            for p in el_vals:
                self.saved_path.append((self.azimuth[y_range_min,curr_y],p))

            self.el_array.append(curr_el)
            self.el_error.append(self.ground_truth[0]-curr_el)

            #Visualization for gradient ascent
            x = range(x_range_min,x_range_max)
            y = len(x)*[curr_x]
            plt.plot(x,y,'o',color='red',markersize=1)

            y = range(y_range_min,y_range_max)
            x = len(y)*[curr_y]
            plt.plot(x,y,'o',color='red',markersize=1)
            

            level = np.linspace(-75,-11,120)
            cs = plt.contourf(plot_signal,levels=level,extend = 'max')
            plt.savefig("./output/GA/fig_{}".format(itr))
            plt.show(block=False)
            plt.pause(1)

            ###########################################
            #Update distance for each iteration
            az_dist = self.get_distance((az_vals[0],curr_el),(az_vals[-1],curr_el))
            el_dist = self.get_distance((curr_az,el_vals[0]),(curr_az,el_vals[-1]))

            self.distance += az_dist + el_dist

            self.dist_array.append(self.distance)

            ############################################
            #Now update current az and el based on gradient
            #print(curr_az,curr_el,delta_az,delta_el)

            print('Current Location:{}'.format((curr_az,curr_el)))
            print('Signal at Current Location:{}'.format(self.signal[curr_x,curr_y]))
            print('Gradient value in Azimuth:{}'.format(delta_az))
            print('Gradient value in Elevation:{}'.format(delta_el))

            next_az = curr_az + (self.alpha*delta_az)
            next_el = curr_el + (self.alpha*delta_el)

            ###########################################
            #Distance update for movement between iterations
            self.distance += self.get_distance((curr_az,curr_el),(next_az,next_el))

            ###########################################

            if (next_az > self.az_max) or (next_az < self.az_min) or (next_el > self.el_max) or (next_el < self.el_min):
                print("Step out of search area!")
                return None

            if (np.absolute(delta_az) < 0.02) and (np.absolute(delta_el) < 0.02):
                maxima = True
            else:
                curr_az = next_az
                curr_el = next_el
            
            #Reverse lookup of index from az and el values
            curr_pos = self.reverse_lookup(curr_az,curr_el)
            curr_x = curr_pos[0]
            curr_y = curr_pos[1]

        self.max_loc = (maxima,curr_x,curr_y)
        print("****************Result*******************")
        print('Location of Maxima:{}'.format((curr_x,curr_y)))
        print('Signal at Maxima:{}'.format(self.signal[curr_x,curr_y]))
        print('Gradient value in Azimuth:{}'.format(delta_az))
        print('Gradient value in Elevation:{}'.format(delta_el))
        print('Total distance for search:{}'.format(self.distance))
        print("*****************************************")


    def get_distance(self,p1,p2):
        az1 = np.deg2rad(p1[0])
        el1 = np.deg2rad(p1[1])
        az2 = np.deg2rad(p2[0])
        el2 = np.deg2rad(p2[1])
        dist = self.ff_radius * np.sqrt(2) * np.sqrt(1 - np.sin(az1)*np.sin(az2)*np.cos(el1-el2) - np.cos(az1)*np.cos(az2))
        return dist

    def get_total_raster_distance(self):
        shape = self.azimuth.shape
        start_point = (int(shape[0]/2),0)
        end_point = (int(shape[0]/2),shape[1]-1)
        row_dist = self.get_distance(start_point,end_point)
        return row_dist*shape[0]

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
        return val,(row,col)


raster_file = "./input/FID1110_07_raster.csv"
rt_data = convert_file2_matrix(raster_file,1300)

signal = rt_data[2,:,:]
area = signal.shape

#Gradient Ascent class instance
grad = gradient_ascent(rt_data)
peak = grad.confirm_maxima()
peak = (rt_data[0,peak[1][0],peak[1][1]],rt_data[1,peak[1][0],peak[1][1]])
grad.find_maxima((int(area[0]/4),int(area[1]/4)))
if grad.max_loc == None:
    print("Search inconclusive!\nSpecify different initial point.")
detected_peak = (grad.max_loc[1],grad.max_loc[2])
detected_peak = (rt_data[0,detected_peak[0],detected_peak[1]],rt_data[1,detected_peak[0],detected_peak[1]])


print(len(grad.el_array))
print(len(grad.dist_array))
plt.clf()
plt.plot(grad.dist_array,grad.el_array,label = "Main Beam (Elevation)")
plt.plot(grad.dist_array,grad.el_error,label = "Elevation Error (from Ground Truth)")
plt.legend()
plt.title("Gradient Ascent (Elevation)")
plt.xlabel("meters")
plt.ylabel("degrees")
plt.show()
# text_file = open('./output/GA/grad_path.txt','w+')
# for p in grad.saved_path:
#     text_file.write(str(p)+'\n')
# text_file.close()

# x = [p[0] for p in grad.saved_path]
# y = [p[1] for p in grad.saved_path]
# plt.clf()
# plt.plot(x,y)
# plt.show()