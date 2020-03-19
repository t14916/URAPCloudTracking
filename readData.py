from netCDF4 import Dataset
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.metrics as skmet
import numpy as np
import math


def readDLData(date, time):
    """
    :param date: Provide date in the following format as a string: YearDayMonth. Ex: 20180504 -> 5-4-2018
    :param time: Provide time in the following format as a string: HourMinuteSecond Ex: 113000 -> 11:30
    :return: dictionary with the following structure:
    "attenuated_backscatter" : used to identify cloud height
    "range" : height which the DL return belongs to
    "radial_velocity" : 2D array of velocities
    "cloud" : 2D array with tuple data of the height and velocity of the clouds
    """
    filename = "sgpdlfptC1.b1." + str(date) + "." + str(time) + ".cdf"
    f = Dataset(filename, "r", format="NETCDF4")
    return_dict = {}

    time = f.variables['time'][:]
    range_var = enumerate(f.variables['range'][:])
    range_var = [r for r in range_var if 500 < r[1] < 5000]

    min_range_index = range_var[0][0] - 1
    max_range_index = range_var[-1][0]
    attenuated_backscatter = f.variables['attenuated_backscatter'][:]
    attenuated_backscatter = [
        [0 if y[i] <= 6e-5 or (min_range_index >= i or i >= max_range_index) else y[i] for i in range(len(y))]
        for y in attenuated_backscatter]
    radial_velocity = f.variables['radial_velocity'][:]

    cloud = []
    removed_rows = []
    for i in range(len(attenuated_backscatter)):
        cloud_row = []
        for j in range(len(attenuated_backscatter[i])):
            if attenuated_backscatter[i][j] != 0:
                cloud_row.append((range_var[j - min_range_index][1], radial_velocity[i][j - min_range_index]))
        if cloud_row:
            cloud.append(cloud_row)
        else:
            removed_rows.append(i)

    adjusted_time = []
    for i in range(len(time)):
        if i not in removed_rows:
            adjusted_time.append(time[i])
    # attenuated_backscatter = [[(range_var[i][1], y[i]) for i in y if y[i] != 0] for y in attenuated_backscatter]
    # range_var = [r[1] for r in range_var]

    # return_dict['attenuated_backscatter'] = attenuated_backscatter
    return_dict['time'] = adjusted_time
    return_dict['radial_velocity'] = radial_velocity
    return_dict['range'] = range_var
    return_dict['cloud'] = cloud

    f.close()
    return return_dict

def clusterClouds(date, time_ld, time_sonde):
    data_ld = readDLData(date, time_ld)
    data_sonde = readSondeData(date, time_sonde)

    plt.ylabel('Height (m)')
    plt.xlabel('Time (h)')
    y = [item[0] for sublist in data_ld["cloud"] for item in sublist]
    rv = [item[1] for sublist in data_ld["cloud"] for item in sublist]

    x = []
    for i in range(len(data_ld["time"])):
        for _ in range(len(data_ld["cloud"][i])):
            x.append(data_ld["time"][i])

    u_wind = []
    v_wind = []
    for i in range(len(y)):
        index = find_approx_value_index(data_sonde["altitude"], y[i], 5)
        if index == -1:
            print("failed to find range")
        u_wind.append(data_sonde["u_wind"][index])
        v_wind.append(data_sonde["v_wind"][index])

    velocity = []
    for i in range(len(u_wind)):
        velocity.append(math.sqrt(u_wind[i] ** 2 + v_wind[i] ** 2))

    # print(velocity)
    # 100m separation using speed
    t_s = [200 / v for v in velocity]

    # 300m mininum length
    t_ll = [300 / v for v in velocity]

    # 5000m maximum length
    t_lh = [5000 / v for v in velocity]

    clusters = map_clusters(y, data_sonde["altitude"], x, t_s, t_ll, t_lh)
    groups = condense_common_values(clusters)

    condensed_x = []
    condensed_y = []
    condensed_rv = []
    condensed_clusters = []
    for i in range(len(clusters)):
        if clusters[i] != 0:
            condensed_x.append(x[i])
            condensed_y.append(y[i])
            condensed_rv.append(rv[i])
            condensed_clusters.append(clusters[i])

    temp = zip(condensed_clusters, condensed_x, condensed_y, condensed_rv)
    #print(temp)
    #print(condensed_rv)
    cloud_data = []
    for i in range(len(groups)):
        cloud_data.append([])
        for t in temp:
            if t[0] != groups[i]:
                break
            cloud_data[i].append(t[1:])

    # clouds provided as separate lists, in time order
    # clouds formatted as tuples with (time (s), altitude (m), radial_velocity (m/s))
    return cloud_data


def condense_common_values(list):
    condensed = []
    for x in list:
        if x not in condensed:
            condensed.append(x)
    return condensed
def plotCloud(date, time_ld, time_sonde):
    # threshold separation = 100m
    # threshold valid cloud > 300m -> 5000m
    # v = sqrt(u^2 + v^2)
    # take radiosonde closest to the above time
    data_ld = readDLData(date, time_ld)
    data_sonde = readSondeData(date, time_sonde)

    plt.ylabel('Height (m)')
    plt.xlabel('Time (h)')
    y = [item[0] for sublist in data_ld["cloud"] for item in sublist]

    x = []
    for i in range(len(data_ld["time"])):
        for _ in range(len(data_ld["cloud"][i])):
            x.append(data_ld["time"][i] / 3600)

    u_wind = []
    v_wind = []
    for i in range(len(y)):
        index = find_approx_value_index(data_sonde["altitude"], y[i], 5)
        if index == -1:
            print("failed to find range")
        u_wind.append(data_sonde["u_wind"][index])
        v_wind.append(data_sonde["v_wind"][index])

    velocity = []
    for i in range(len(u_wind)):
        velocity.append(math.sqrt(u_wind[i]**2 + v_wind[i]**2))

    #print(velocity)
    #100m separation using speed
    t_s = [200 / v for v in velocity]

    #300m mininum length
    t_ll = [300 / v for v in velocity]

    #5000m maximum length
    t_lh = [5000 / v for v in velocity]

    #print(t_s)
    clusters = map_clusters(y, data_sonde["altitude"], [3600 * element for element in x], t_s, t_ll, t_lh)
    #print(len(clusters))
    #print(clusters)

    # KMeans code: NOTE -> does not work that well with time in hours
    #x_arr = np.asarray(x)
    #y_arr = np.asarray(y)
    #x_arr = x_arr.reshape(-1, 1)
    #y_arr = y_arr.reshape(-1, 1)
    #k_arr = np.hstack((x_arr, y_arr))
    #k = optimal_k(k_arr, 10)
    #kmeans = KMeans(n_clusters=k).fit(k_arr)

    #print(x)
    x_plot = []
    y_plot = []
    clusters_plot = []
    for i in range(len(clusters)):
        if clusters[i] != 0:
            x_plot.append(x[i])
            y_plot.append(y[i])
            clusters_plot.append(clusters[i])

    #print(clusters_plot)
    plt.scatter([3600*element for element in x], y, c=clusters, cmap='rainbow', marker='.')  #, c=kmeans.labels_, cmap='rainbow')
    #plt.scatter(x_plot, y_plot, c=clusters_plot, cmap='rainbow')
    plt.show()
    return 0


def map_clusters(cloud_height, altitude, time, t_s, t_ll, t_lh):
    color_map = [1]
    curr_cluster = 1
    t_start = 0
    # print(time[0], time[-1])
    # print(cloud_height)
    for i in range(1, len(time)):
        # print("({}, {})".format(time[i], cloud[i]))
        thresh_index = find_approx_value_index(altitude, cloud_height[i], 5)
        if time[i] - time[i - 1] < t_s[thresh_index] and time[i] - time[t_start] < t_lh[thresh_index]:
             color_map.append(curr_cluster)
        else:
            # print("time before: {} time after: {}".format(time[i-1], time[i]))
            # print(i)
            # print("time seperation: {} threshold: {}".format(time[i] - time[i - 1], t_s[i]))
            # print("cloud length: {} min length: {} max length: {}".format(time[i - 1] - time[t_start], t_ll[i], t_lh[i]))
            if time[i - 1] - time[t_start] < t_ll[thresh_index]:
                curr_cluster -= 1
                for j in range(t_start, i):
                    color_map[j] = 0
            curr_cluster += 1
            color_map.append(curr_cluster)
            t_start = i
    return color_map

def find_approx_value_index(list, value, threshold):
    for i in range(len(list)):
        if value - threshold < list[i] < value + threshold:
            return i
    return -1

def optimal_k(points, kmax):
    # Using silhouette score to determine optimal k value for kmeans
    sil = []

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(points)
        labels = kmeans.labels_
        sil.append(skmet.silhouette_score(points, labels, metric='euclidean'))

    return sil.index(max(sil)) + 2


def readSondeData(date, time):
    """
    :param date: Provide date in the following format as a string: YearDayMonth. Ex: 20180504 -> 5-4-2018
    :param time: Provide time in the following format as a string: HourMinuteSecond Ex: 113000 -> 11:30
    :return: dictionary with the following structure:
    "u_wind" : wind speed in u direction (x direction)
    "v_wind" : wind speed in v direction (y direction)
    "alt" : altitude of cloud
    """
    filename = "sgpsondewnpnC1.b1." + str(date) + "." + str(time) + ".cdf"
    f = Dataset(filename, "r", format="NETCDF4")
    # print(f.variables)
    return_dict = {}
    temp = f.variables['alt'][:]
    removed_indices = []
    altitude = []
    for i in range(len(temp)):
        a = temp[i]
        if 500 < a < 5000:
            altitude.append(a)
        else:
            removed_indices.append(i)

    u_temp = f.variables['u_wind'][:]
    v_temp = f.variables['v_wind'][:]
    u_wind = []
    v_wind = []
    for i in range(len(u_temp)):
        if i not in removed_indices:
            u_wind.append(u_temp[i])
            v_wind.append(v_temp[i])

    return_dict['u_wind'] = u_wind
    return_dict['v_wind'] = v_wind
    return_dict['altitude'] = altitude
    f.close()

    return return_dict


# Center of COGS data is where the LIDAR is positioned, so we can see the LIDAR data along with the COGS data
def readCOGSData():
    return 0

date = 20180504
time_ld = 200117
#time_ld = 210116
time_sonde = 233600
#plotCloud(date, time_ld, time_sonde)
temp = clusterClouds(date, time_ld, time_sonde)
print(temp)
