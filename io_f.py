# coding: utf-8
"""
Store the wifi ssids as a set for each building
"""
from collections import defaultdict
from dataclasses import dataclass
from building import BUILDING_INFO
import numpy as np

@dataclass
class ReadData:
    acce: np.ndarray
    acce_uncali: np.ndarray
    gyro: np.ndarray
    gyro_uncali: np.ndarray
    magn: np.ndarray
    magn_uncali: np.ndarray
    ahrs: np.ndarray
    wifi: np.ndarray
    ibeacon: np.ndarray
    waypoint: np.ndarray
    test_mode: bool = False

    def wifi_data_to_dict(self, use_avg=False):
        """
        read the wifi data into a dictionary
        """
        wifi_dict = defaultdict(lambda: {
            'freq': set(),
            "delta_t": 0,
        })

        if use_avg:
            measured_times = defaultdict(list)  # reported_time : [measured_times..]
        else:
            measured_times = defaultdict(lambda: [0])
        for iwd in self.wifi:
            t0, ssids, _, _, freq, t1 = iwd
            t_out, x_out, y_out = get_pos_by_time(np.array([t0]), self.waypoint) # check if the point can be interpolated on
            if not self.test_mode and (len(t_out) == 0 or freq[0] not in ["2", "5"]):
                continue
            if use_avg:
                measured_times[t0].append(t1)
            else:
                measured_times[t0][0] = t0
            wifi_dict[(t0, ssids)]["freq"].add(int(freq[0]))
            wifi_dict[(t0, ssids)]["delta_t"] = max(wifi_dict[(t0, ssids)]["delta_t"], t0 - t1)

        return wifi_dict, measured_times

    def get_wifi_data_array(self, building_id, use_avg: bool = False):
        """
        For a dataset in a building convert the valid wifi data at each t0 into a 1-hot encoded data
        """
        freq_map = {
            (2,): 1,
            (5,): 2,
            (2, 5): 3,
        }
        ssid_encoding = BUILDING_INFO[building_id]['wifi_ssids']
        wifi_dict, measured_times = self.wifi_data_to_dict(use_avg=use_avg)
        times = sorted(list({t for t, _ in wifi_dict.keys()}))
        t_index = {v: i for i, v in enumerate(times)}
        wifi_data = np.zeros((len(times), len(ssid_encoding), 2))

        for (t, ssid), d in wifi_dict.items():
            if ssid not in ssid_encoding:
                continue
            freq = tuple(sorted(d['freq']))
            wifi_data[t_index[t]][ssid_encoding[ssid]][0] = freq_map[freq]
            wifi_data[t_index[t]][ssid_encoding[ssid]][1] = d['delta_t']
        return wifi_data, measured_times

    def get_positions_at_measured_times(self, measured_times):
        """
        Fore a measured_times dictionary that has {reported_time, [measured_times]}
        calculate the position that correponds to that reported_time
        Args:
            measured_times: measured_times dictionary

        Returns:
            Array that has n X 3 data where each row is (reported_time, x, y)

        """
        times = sorted(list(measured_times.keys()))
        t_index = {v: i for i, v in enumerate(times)}
        p_out = np.zeros((len(times), 3))
        for tt, m_times in measured_times.items():
            t_avg, x_avg, y_avg = get_pos_by_time(np.array(m_times), self.waypoint)
            p_out[t_index[tt],:] = np.array([int(tt), x_avg.mean(), y_avg.mean()], dtype=np.float)
        return p_out

    def get_interpolated_sensor_data(self, times, sensor_names = ['magn', 'acce', 'gyro']):
        """
        For a list of time stamps, concat all the sensor data together in sequence
        following the order of sensor_names
        Returns:
            Array len(times) x sum of all sensor data dimensions
        """
        res = np.array(sorted(times), dtype=np.float)
        res = np.expand_dims(res, 1)
        for sn in sensor_names:
            s_data = getattr(self, sn)
            for dim in range(1, s_data.shape[1]):
                new_data = np.interp(res[:,0], s_data[:, 0], s_data[:, dim])
                new_data = np.expand_dims(new_data, 1)
                res = np.concatenate([res, new_data], axis=1)
        return res

    def get_training_data(self, building_id, use_avg: bool, sensor_names: list):
        """
        REturn the full set of training data
        Args:
            use_avg: use the measure time averaged position
            sensor_names: the list of sensor names to use

        Returns:
            array representing the input and output of the training data
        """
        if self.test_mode:
            raise RuntimeError("Must use training data we need to interpolated for positions")

        wifi_data, mt = self.get_wifi_data_array(building_id=building_id, use_avg=use_avg)
        sensor_data = self.get_interpolated_sensor_data(mt.keys(), sensor_names=sensor_names)
        positions = self.get_positions_at_measured_times(mt)
        return wifi_data, sensor_data, positions

    def get_test_data(self, building_id, sensor_names: list):
        """
        Return the full vector that represents the test data
        Args:
            sensor_names: the list of sensor names to use

        Returns:
            array representing the input for the test
        """
        if not self.test_mode:
            raise RuntimeError("Must use testing data")

        wifi_data, mt = self.get_wifi_data_array(building_id=building_id, use_avg=False)
        sensor_data = self.get_interpolated_sensor_data(mt.keys(), sensor_names=sensor_names)
        return wifi_data, sensor_data


def read_data_file(data_filename, test_mode=False):
    acce = []
    acce_uncali = []
    gyro = []
    gyro_uncali = []
    magn = []
    magn_uncali = []
    ahrs = []
    wifi = []
    ibeacon = []
    waypoint = []

    with open(data_filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line_data in lines:
        line_data = line_data.strip()
        if not line_data or line_data[0] == '#':
            continue

        line_data = line_data.split('\t')

        if line_data[1] == 'TYPE_ACCELEROMETER':
            acce.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_ACCELEROMETER_UNCALIBRATED':
            acce_uncali.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_GYROSCOPE':
            gyro.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_GYROSCOPE_UNCALIBRATED':
            gyro_uncali.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_MAGNETIC_FIELD':
            magn.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_MAGNETIC_FIELD_UNCALIBRATED':
            magn_uncali.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        # if line_data[1] == 'TYPE_ROTATION_VECTOR':
        #     ahrs.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
        #     continue

        if line_data[1] == 'TYPE_WIFI':
            sys_ts = int(line_data[0])
            ssid = line_data[2]
            bssid = line_data[3]
            rssi = line_data[4]
            freq = line_data[5]
            lastseen_ts = int(line_data[6])
            wifi_data = [sys_ts, ssid, bssid, rssi, freq, lastseen_ts]
            wifi.append(wifi_data)
            continue

        if line_data[1] == 'TYPE_BEACON':
            ts = line_data[0]
            uuid = line_data[2]
            major = line_data[3]
            minor = line_data[4]
            rssi = line_data[6]
            ibeacon_data = [ts, '_'.join([uuid, major, minor]), rssi]
            ibeacon.append(ibeacon_data)
            continue

        if line_data[1] == 'TYPE_WAYPOINT':
            waypoint.append([int(line_data[0]), float(line_data[2]), float(line_data[3])])
    waypoint.sort()
    wifi.sort()

    magn.sort()
    acce.sort()
    gyro.sort()
    magn_uncali.sort()

    magn = np.array(magn)
    acce = np.array(acce)
    gyro = np.array(gyro)
    magn_uncali = np.array(magn_uncali)

    return ReadData(acce, acce_uncali, gyro, gyro_uncali, magn, magn_uncali, ahrs, wifi, ibeacon, waypoint, test_mode=test_mode)

def get_pos_by_time(t, way_points):
    # Read the way points at time then perform a inverse square weighted average 
    # to approximate the fact time people spend more time near the measurements
    x = np.ones_like(t)*-1E10
    y = np.ones_like(t)*-1E10
    for i in range(len(way_points)-1):
        t1, x1, y1 = way_points[i]
        t2, x2, y2 = way_points[i+1]
        mask = (t >= t1) * (t < t2)
        if any(mask):
            tt = t[mask]
            f1 = np.abs(tt-t1)/(t2-t1)
            f2 = np.abs(tt-t2)/(t2-t1)
            f1, f2 = 1/f1**2, 1/f2**2
            f1, f2 = f1/(f1 + f2), f2/(f1 + f2)
            f2 = np.nan_to_num(f2, copy=False, nan=1, posinf=1, neginf=0)
            f1 = np.nan_to_num(f1, copy=False, nan=1, posinf=1, neginf=0)
            x[mask] = f1 * x1 + f2 * x2
            y[mask] = f1 * y1 + f2 * y2
    res_mask = (x > -1E10) * (y > -1E10)
    return t[res_mask], x[res_mask], y[res_mask]

