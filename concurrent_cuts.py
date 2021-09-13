import numpy as np 
import csv
import sys
import matplotlib.pyplot as plt

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

class concurrent_cuts():
    def __init__(self,rt_data):
        self.rt_data = rt_data
        self.signal = rt_data[2,:,:]
        self.azimuth = rt_data[0,:,:]
        self.elevation = rt_data[1,:,:]
        self.distance = 0
        self.ff_radius = 100
        self.saved_path = []
        self.az_min,self.az_max = self.find_ranges(self.azimuth) 
        self.el_min,self.el_max = self.find_ranges(self.elevation.transpose())

        _,peak = self.confirm_maxima()
        self.ground_truth = (rt_data[0,peak[0],peak[1]],rt_data[1,peak[0],peak[1]])
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
                curr_x = i

        min_diff = np.absolute(el - self.elevation[0,0])
        for i in range(self.elevation.shape[0]):
            diff = np.absolute(el - self.elevation[i,0])
            if diff < min_diff:
                min_diff = diff
                curr_y = i

        return curr_x,curr_y

    def find_maxima(self,az_pos,el_pos):
        plot_signal = np.zeros(self.signal.shape)
        plot_signal[:] = self.signal
        
        maxima = False
        itr = 0

        shape = self.azimuth.shape
        scan_angle = 8
        while(not(maxima)):
            print("Itr:{}".format(itr))
            az_index,el_index = self.reverse_lookup(az_pos,el_pos)

            plt.plot(az_index,el_index,'o',color='magenta',markersize=6)
            #For fixed elevation
            curr_az_begin = az_pos - scan_angle
            curr_az_end = az_pos + scan_angle
            if curr_az_begin < self.az_min:
                curr_az_begin = self.az_min
            if curr_az_end > self.az_max:
                curr_az_end = self.az_max
            curr_az_begin_index,_ = self.reverse_lookup(curr_az_begin,el_pos)
            curr_az_end_index,_ = self.reverse_lookup(curr_az_end,el_pos)
            
            az_signal_vals = self.signal[el_index,curr_az_begin_index:curr_az_end_index]

            az_max_index = np.argmax(az_signal_vals) +  curr_az_begin_index
            
            #Plot for fixed elevation
            x = range(curr_az_begin_index,curr_az_end_index+1)
            y = len(x) * [el_index]
            plt.plot(x,y,'*',color='red',markersize=0.5)

            

            #For fixed azimuth
            curr_el_begin = el_pos - scan_angle
            curr_el_end = el_pos + scan_angle
            if curr_el_begin < self.el_min:
                curr_el_begin = self.el_min
            if curr_el_end > self.el_max:
                curr_el_end = self.el_max
            _,curr_el_begin_index = self.reverse_lookup(az_pos,curr_el_begin)
            _,curr_el_end_index = self.reverse_lookup(az_pos,curr_el_end)
            
            el_signal_vals = self.signal[curr_el_begin_index:curr_el_end_index,az_index]
            
            el_max_index = np.argmax(el_signal_vals) + curr_el_begin_index
            
            
            self.saved_path.append((curr_az_begin,el_pos))
            self.saved_path.append((curr_az_end,el_pos))
            self.saved_path.append((az_pos,curr_el_begin))
            self.saved_path.append((az_pos,curr_el_end))

            #Plot for fixed azimuth
            y = range(curr_el_begin_index,curr_el_end_index+1)
            x = len(y)*[az_index]
            plt.plot(x,y,'*',color='red',markersize=1)

            plt.plot(az_max_index,el_max_index,'o',color='red',markersize=6)
            
            

            level = np.linspace(-75,-11,120)
            cs = plt.contourf(plot_signal,levels=level,extend = 'max')
            plt.savefig("./output/CC/fig_{}".format(itr))
            itr += 1
            plt.xlabel("Azimuth")
            plt.ylabel("Elevation")
            plt.show(block=False)
            plt.pause(2)
            plt.clf()

            current_az_max = self.azimuth[el_max_index,az_max_index]
            current_el_max = self.elevation[el_max_index,az_max_index]
            self.az_array.append(current_az_max)
            self.az_error.append(self.ground_truth[0]-current_az_max)
            self.el_array.append(current_el_max)
            self.el_error.append(self.ground_truth[1]-current_el_max)

            #Re-assign for next search
            if int(current_az_max/az_pos) == 1 and int(current_el_max/el_pos) == 1:
                dist_1 = self.get_distance((curr_az_begin,el_pos),(curr_az_end,el_pos))
                dist_2 = self.get_distance((curr_az_end,el_pos),(az_pos,curr_el_begin))
                dist_3 = self.get_distance((az_pos,curr_el_begin),(az_pos,curr_el_end))
                self.distance += (dist_1+dist_2+dist_3)
                maxima = True
            else:
                if itr == 1:
                    dist_1 = self.get_distance((curr_az_begin,el_pos),(curr_az_end,el_pos))
                    dist_2 = self.get_distance((curr_az_end,el_pos),(az_pos,curr_el_begin))
                    dist_3 = self.get_distance((az_pos,curr_el_begin),(az_pos,curr_el_end))
                    previous_az_pos = az_pos
                    previous_el_end = curr_el_end
                    self.distance += (dist_1+dist_2+dist_3)
                else:
                    dist_1 = self.get_distance((curr_az_begin,el_pos),(curr_az_end,el_pos))
                    dist_2 = self.get_distance((curr_az_end,el_pos),(az_pos,curr_el_begin))
                    dist_3 = self.get_distance((az_pos,curr_el_begin),(az_pos,curr_el_end))
                    dist_4 = self.get_distance((previous_az_pos,previous_el_end),(curr_az_begin,el_pos))
                    self.distance += (dist_1+dist_2+dist_3+dist_4)
                az_pos = current_az_max
                el_pos = current_el_max

            self.dist_array.append(self.distance)
            if itr<4:
                scan_angle = scan_angle/2
            if itr == 8:
                break

                

        print("************Result************")
        print('Location of Maxima in Azimuth:{}'.format(current_az_max))
        print('Location of Maxima in Elevation:{}'.format(current_el_max))
        print('Total Search Distance:{}'.format(self.distance))
        print("******************************")
        return (current_az_max,current_el_max)
    
    def get_distance(self,p1,p2):
        az1 = np.deg2rad(p1[0])
        el1 = np.deg2rad(p1[1])
        az2 = np.deg2rad(p2[0])
        el2 = np.deg2rad(p2[1])
        dist = self.ff_radius * np.sqrt(2) * np.sqrt(1 - np.cos(az1)*np.cos(az2)*np.cos(el1-el2) - np.sin(az1)*np.sin(az2))
        return dist

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

#instance of class
conc = concurrent_cuts(rt_data)
peak = conc.confirm_maxima()
peak = (rt_data[0,peak[1][0],peak[1][1]],rt_data[1,peak[1][0],peak[1][1]])

detected_maxima = conc.find_maxima(3.5,3.5)

text_file = open('./output/CC/conc_path.txt','w+')
for p in conc.saved_path:
    text_file.write(str(p)+'\n')
text_file.close()

# x = [p[0] for p in conc.saved_path]
# y = [p[1] for p in conc.saved_path]
# plt.clf()
# plt.plot(x,y)
# plt.show()

print(len(conc.az_array))
print(len(conc.dist_array))
plt.clf()
plt.step(conc.dist_array,conc.az_array,label = "Main Beam (Azimuth)")
plt.step(conc.dist_array,conc.az_error,label = "Azimuth Error (from Ground Truth)")
plt.legend()
plt.title("Concurrent Cuts (Azimuth)")
plt.xlabel("meters")
plt.ylabel("degrees")
# plt.ylim((-4,4))
plt.show()