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

    qs_file = open(file_name)
    csv_file = csv.reader(qs_file,delimiter=',')
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

    qs_file.close()
    return rt_data

class maxima_search():
    def __init__(self,rt_data,start):
        self.rt_data = rt_data
        self.start = start
        self.step_angle = 10
        self.rad_max = 40
        self.rad_min = 5
        self.x_max = rt_data[2,:,:].shape[0]
        self.x_min = 0
        self.y_max = rt_data[2,:,:].shape[1]
        self.y_min = 0
        self.ff_radius = 100
        self.dist = 0
        self.az_min,self.az_max = self.find_ranges(self.rt_data[0,:,:])
        self.el_min,self.el_max = self.find_ranges(self.rt_data[1,:,:].transpose())
        self.azimuth = rt_data[0,:,:]
        self.elevation = rt_data[1,:,:]
        self.saved_path= []

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

    def get_path_cricular2(self,center,rad,step):
        path = []
        start = (center[0] + rad, center[1])

        angles = np.linspace(0,360,num = int(360/step)+1)
        for ang in angles:
            curr_x = (rad*np.cos(np.deg2rad(ang)) + center[0])
            curr_y = (rad*np.sin(np.deg2rad(ang)) + center[1])

            if (curr_x >= self.az_max) or (curr_x < self.az_min):
                continue
            if (curr_y >= self.el_max) or (curr_y < self.el_min):
                continue
            path.append((curr_x,curr_y))

        return path

    def reverse_lookup(self,az,el):
        curr_x = 0
        curr_y = 0

        #To conduct search at the center of the data
        #instead of the edge to avoid any errors
        search_line_az = int(self.azimuth.shape[0]/2)
        search_line_el = int(self.azimuth.shape[1]/2)

        min_diff = np.absolute(az - self.azimuth[0,0])
        for i in range(self.azimuth.shape[1]):
            #diff = np.absolute(az - self.azimuth[0,i])
            diff = np.absolute(az - self.azimuth[search_line_az,i])
            if diff < min_diff:
                min_diff = diff
                curr_y = i

        min_diff = np.absolute(el - self.elevation[0,0])
        for i in range(self.elevation.shape[0]):
            #diff = np.absolute(el - self.elevation[i,0])
            diff = np.absolute(el - self.elevation[i,search_line_el])
            if diff < min_diff:
                min_diff = diff
                curr_x = i

        return (curr_x,curr_y)

    def convert_angular_path_to_coord(self,path):
        temp = []
        for p in path:
            temp.append(self.reverse_lookup(p[0],p[1]))
        return temp

    def find_peak(self,radius,center):
        signal = self.rt_data[2,:,:]
        plot_signal = np.zeros(signal.shape)
        plot_signal2 = np.zeros(signal.shape)
        plot_signal2[:] = signal
        maxima = False

        curr_x = center[0]
        curr_y = center[1]

        center_signal_val = signal[center[0],center[1]]
        curr_x = center[0]
        curr_y = center[1]

        self.saved_path.append((self.rt_data[0,curr_x,curr_y],self.rt_data[1,curr_x,curr_y]))
        k = 0
        radii = np.linspace(radius,0.2,5)
        for curr_radius in radii:
            maxima = False
            while(not(maxima)):
                print("Itr!")
                k = k+1
                path = self.get_path_cricular2((self.rt_data[0,curr_x,curr_y],self.rt_data[1,curr_x,curr_y]),curr_radius,self.step_angle)
                for p in path:
                    self.saved_path.append(p)
                
                path = self.convert_angular_path_to_coord(path)
                
                self.dist += self.get_search_distance((curr_x,curr_y),path)
                
                #Now go through all the signal values in path
                max_signal_val = signal[path[0][0],path[0][1]]
                max_signal_pos = (path[0][0],path[0][1])
                for p in path:
                    curr_signal_val = signal[p[0],p[1]]
                    if curr_signal_val > max_signal_val:
                        max_signal_val = curr_signal_val
                        max_signal_pos = (p[0],p[1])
                
                print('Max Signal on circumference:{} , Signal at center:{}'.format(max_signal_val,center_signal_val))
                if max_signal_val < center_signal_val:
                    maxima = True
                    max_signal_pos = (curr_x,curr_y)
                    max_signal_val = center_signal_val

                else:
                    curr_x = max_signal_pos[0]
                    curr_y = max_signal_pos[1]
                    center_signal_val = max_signal_val
                
                
                print(max_signal_pos)
                ###This part is for visualizing maxima search
                plot_signal[:] = signal
                for p in path:
                    plot_signal[p[0],p[1]] = 1
                plot_signal[max_signal_pos[0],max_signal_pos[1]]=1

                y = [p[0] for p in path]
                x = [p[1] for p in path]
                plt.plot(x,y,'o',color='red',markersize=1)
                
                
                level = np.linspace(-75,-11,120)
                cs = plt.contourf(plot_signal2,levels=level,extend = 'max')
                
                plt.savefig("./output/DFS/fig_{}".format(k))
                plt.show(block=False)
                plt.pause(0.5)
                plot_signal = np.zeros(signal.shape)
                plot_signal[:] = signal
                ###
            #return max_signal_val,max_signal_pos,self.dist
        return max_signal_val,max_signal_pos,self.dist

    def find_beam_width1(self,max_signal_val,max_signal_pos):
        signal = self.rt_data[2,:,:]
        center = max_signal_pos
        halfpower = False #Flag indicating half power point found or not
        halfpower_val = max_signal_val - 3 #This represents -3 db fall in signal level
        direction = ['up','down','right','left']
        direction_coordinates = []
        for d in direction:
            halfpower = False 
            if d == 'up':
                next_point_x,next_point_y = center[0],center[1]+1
                while(not(halfpower)):
                    signal_at_next = signal[next_point_x,next_point_y]
                    if signal_at_next <= halfpower_val:
                        halfpower = True
                        direction_coordinates.append((next_point_x,next_point_y))
                    next_point_y = next_point_y + 1
            if d == 'down':
                next_point_x,next_point_y = center[0],center[1]-1
                while(not(halfpower)):
                    signal_at_next = signal[next_point_x,next_point_y]
                    if signal_at_next <= halfpower_val:
                        halfpower = True
                        direction_coordinates.append((next_point_x,next_point_y))
                    next_point_y = next_point_y - 1
            if d == 'right':
                next_point_x,next_point_y = center[0]+1,center[1]
                while(not(halfpower)):
                    signal_at_next = signal[next_point_x,next_point_y]
                    if signal_at_next <= halfpower_val:
                        halfpower = True
                        direction_coordinates.append((next_point_x,next_point_y))
                    next_point_x = next_point_x + 1
            if d == 'left':
                next_point_x,next_point_y = center[0]-1,center[1]
                while(not(halfpower)):
                    signal_at_next = signal[next_point_x,next_point_y]
                    if signal_at_next <= halfpower_val:
                        halfpower = True
                        direction_coordinates.append((next_point_x,next_point_y))
                    next_point_x = next_point_x - 1

        halfpower_az1 = direction_coordinates[3]
        halfpower_az2 = direction_coordinates[2]

        halfpower_el1 = direction_coordinates[1]
        halfpower_el2 = direction_coordinates[0]

        #####################
        print(direction_coordinates)
        # print(self.rt_data[0,halfpower_az2[0],halfpower_az2[1]])
        # print(self.rt_data[0,halfpower_az1[0],halfpower_az1[1]])

        # print(self.rt_data[2,halfpower_az2[0],halfpower_az2[1]])
        # print(self.rt_data[2,halfpower_az1[0],halfpower_az1[1]])

        #####################
        beam_width_az = self.rt_data[1,halfpower_az2[0],halfpower_az2[1]] - self.rt_data[1,halfpower_az1[0],halfpower_az1[1]]
        beam_width_el = self.rt_data[0,halfpower_el2[0],halfpower_el2[1]] - self.rt_data[0,halfpower_el1[0],halfpower_el1[1]]
        return beam_width_az,beam_width_el

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
    
    def get_search_distance(self,center,path):
        start_point = path[0]
        dist = self.get_distance(center,start_point)
        #This is the formula for circumference + one radisu
        path_dist = (2 * np.pi * dist) + dist
        return path_dist


    def get_distance(self,p1,p2):
        az1 = np.deg2rad(self.rt_data[0,p1[0],p1[1]])
        el1 = np.deg2rad(self.rt_data[1,p1[0],p1[1]])
        az2 = np.deg2rad(self.rt_data[0,p2[0],p2[1]])
        el2 = np.deg2rad(self.rt_data[1,p2[0],p2[1]])
        #print(az1,el1,az2,el2)
        dist = self.ff_radius * np.sqrt(2) * np.sqrt(1 - np.sin(az1)*np.sin(az2)*np.cos(el1-el2) - np.cos(az1)*np.cos(az2))
        return dist

    def get_total_raster_distance(self):
        shape = self.azimuth.shape
        start_point = (int(shape[0]/2),0)
        end_point = (int(shape[0]/2),shape[1]-1)
        row_dist = self.get_distance(start_point,end_point)
        return row_dist*shape[0]

raster_file = "./input/FID1110_07_raster.csv"
rt_data = convert_file2_matrix(raster_file,1300)
signal = rt_data[2,:,:]
area = signal.shape
#DFS search circular

#Class instance
kns = maxima_search(rt_data,(int(area[0]/2),int(area[1]/2)))


max_signal_val,max_signal_pos,dist = kns.find_peak(1,(int(area[0]/4),int(area[1]/4)))
print('Max signal value found:{}'.format(max_signal_val))
print('Location of Max Signal:{}'.format(max_signal_pos))
print('Distance covered for search:{}'.format(dist))
print("Total Raster Distance:{}".format(kns.get_total_raster_distance()))
peak = kns.confirm_maxima()


text_file = open('./output/DFS/DFS_path.txt','w+')
for p in kns.saved_path:
    text_file.write(str(p)+'\n')
text_file.close()

x = [p[0] for p in kns.saved_path]
y = [p[1] for p in kns.saved_path]
plt.clf()
plt.plot(x,y)
plt.show()