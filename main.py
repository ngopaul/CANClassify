from utils import *
import random
import pickle
from copy import copy, deepcopy
from datetime import datetime
import random
random.seed(0)

# disable the gpu
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
from keras import layers


labels = [
    'velocity_x', # velocity in the forward direction
    'acceleration_x', # acceleration in the forward direction
    'acceleration_y', # acceleration in the horizontal direction
    'steer_angle',
    'steer_angle_rate',
    'brake_pedal', # continuous value for how much it is pressed
    'gas_pedal', # continuous value for how much it is pressed
    'radar_long', # longitudinal
    'radar_lat', # latitudinal
    'radar_rel_vel', # relative velocity in the forward direction
    'radar_rel_acc', # relative acceleration in the forward direction
    'checksum',
]

# translate from our naming system to the message/signal name for each vehicle. This needs to be done for each message/signal
# pair we wish to train on

label_to_messig_toyota = {
    'velocity_x': [
        ('WHEEL_SPEEDS', 'WHEEL_SPEED_FR'),
        ('WHEEL_SPEEDS', 'WHEEL_SPEED_FL'),
        ('WHEEL_SPEEDS', 'WHEEL_SPEED_RR'),
        ('WHEEL_SPEEDS', 'WHEEL_SPEED_RL')
    ],
    'acceleration_x': [
        ('ACCELEROMETER', 'ACCEL_X')
    ],
    'acceleration_y': [
        ('KINEMATICS', 'ACCEL_Y')
    ],
    'steer_angle': [('STEER_ANGLE_SENSOR', 'STEER_ANGLE')],
    'steer_angle_rate': [('STEER_ANGLE_SENSOR', 'STEER_RATE')],
    'brake_pedal': [('BRAKE', 'BRAKE_PEDAL')],
    'gas_pedal': [('GAS_PEDAL', 'GAS_PEDAL')],
    'radar_long': [
        (f'TRACK_A_{i}', 'LONG_DIST') for i in range(16)
    ],
    'radar_lat': [
        (f'TRACK_A_{i}', 'LAT_DIST') for i in range(16)
    ],
    'radar_rel_vel': [
        (f'TRACK_A_{i}', 'REL_SPEED') for i in range(16)
    ],
    'radar_rel_acc': [
        (f'TRACK_B_{i}', 'REL_ACCEL') for i in range(16)
    ]
}

label_to_messig_honda = {
    'velocity_x': [
        ('WHEEL_SPEEDS', 'WHEEL_SPEED_FR'),
        ('WHEEL_SPEEDS', 'WHEEL_SPEED_FL'),
        ('WHEEL_SPEEDS', 'WHEEL_SPEED_RR'),
        ('WHEEL_SPEEDS', 'WHEEL_SPEED_RL')
    ],
    'acceleration_x': [
        ('VEHICLE_DYNAMICS', 'LONG_ACCEL')
    ],
    'steer_angle': [
        ('STEERING_SENSORS', 'STEER_ANGLE')
    ],
    'steer_angle_rate': [
        ('STEERING_SENSORS', 'STEER_ANGLE_RATE')
    ],
    'brake_pedal': [
        ('POWERTRAIN_DATA', 'BRAKE_PRESSED')
    ],
    'gas_pedal': [
        ('GAS_PEDAL', 'CAR_GAS')
    ]
}

label_to_messig_nissan = {
    'velocity_x': [
        ('WHEEL_SPEEDS_FRONT', 'WHEEL_SPEED_FR'),
        ('WHEEL_SPEEDS_FRONT', 'WHEEL_SPEED_FL'),
        ('WHEEL_SPEEDS_REAR', 'WHEEL_SPEED_RR'),
        ('WHEEL_SPEEDS_REAR', 'WHEEL_SPEED_RL')
    ],
    'steer_angle': [
        ('STEER_ANGLE_SENSOR', 'STEER_ANGLE')
    ],
    'steer_angle_rate': [
        ('STEER_ANGLE_SENSOR', 'STEER_ANGLE_RATE')
    ],
    'brake_pedal': [
        ('BRAKE_PEDAL', 'BRAKE_PEDAL')
    ],
}


class VEHICLE:
    def __init__(self, name, cantools_db, label_to_messig, csv_paths):
        self.name = name
        self.cantools_db = cantools_db
        self.label_to_messig = label_to_messig
        self.csv_paths = csv_paths


def get_vehicle_with_name(name, vehicles):
    for vehicle in vehicles:
        if vehicle.name == name:
            return vehicle
    return None


def get_trajectory_dict(csv_paths, cantools_db, labels, label_to_messig, more_info_to_print=""):
    """Gets a dictionary of raw binary-valued trajectories,
    given csv names with actual drives, desired signals, and the cantools_db for decoding

    Args:
        csv_paths: a list of paths to the csv files with raw drive data
        cantools_db: a cantools db which contains the dbc information of the vehicle
        labels: a list of signals of interest, in human readable form
        label_to_messig: a dictionary which maps from labels to lists of tuples involved
            (message_id, signal_id)

    Returns:
        dictionary of trajectories, mapping key=labels to value=list of xs/ys
        xs are a list of list of timestamps
        ys are a list of list of binary values
        The reason for multiple x and y is because there are multiple buses. len(xs) = len(ys) = number of
            buses which returned valid data
    """
    trajectories = {k: [] for k in labels}
    for csv_path in csv_paths:
        pd_csv = pd.read_csv(csv_path)
        for label in labels:
            if not label in label_to_messig:  # the vehicle doesn't have a known labelling for this signal
                continue

            for message_name, signal_name in label_to_messig[label]:
                # get the encoding for this message/signal
                _, _, value_dict = describe_known_signal(cantools_db, message_name, signal_name, None, None,
                                                         silent=True)
                # get the actual binary values
                raw_xs, raw_ys = get_raw_signal_values(
                    pd_csv, value_dict["frame_id"], value_dict["start"], value_dict["signal_length"],
                    value_dict["byte_order"], bus_limit=1, return_np_bool_array=True)

                if not len(raw_xs):
                    # there is no signal in this file, i.e. no bus provided valid data
                    continue

                raw_xs = raw_xs[0]
                raw_ys = raw_ys[0]

                print(f"Got trajectory w/ {len(raw_ys)} timepoints, csv {csv_path}, "
                      f"{label}:{message_name}/{signal_name}{', ' + more_info_to_print if more_info_to_print else ''}")

                trajectories[label].append((raw_xs, raw_ys))

    return trajectories


def convert_ith_original_signal_to_convolved_signal(y, i):
    new_y = np.zeros(716)

    if not (type(y[i]) == str) and not (type(y[i]) == np.str_):
        y_i = ''.join(y[i].astype(int).astype(str))
    else:
        y_i = y[i]

    for j in range(61):
        new_y[j] = mask_bin_str(y_i[j:j + 4], byteorder='big', signed=False)
    for j in range(61):
        new_y[j + 61] = mask_bin_str(y_i[j:j + 4], byteorder='little', signed=False)

    for j in range(57):
        new_y[j + 122] = mask_bin_str(y_i[j:j + 8], byteorder='big', signed=False)
    for j in range(57):
        new_y[j + 179] = mask_bin_str(y_i[j:j + 8], byteorder='little', signed=False)
    for j in range(57):
        new_y[j + 236] = mask_bin_str(y_i[j:j + 8], byteorder='big', signed=True)
    for j in range(57):
        new_y[j + 293] = mask_bin_str(y_i[j:j + 8], byteorder='little', signed=True)

    for j in range(53):
        new_y[j + 350] = mask_bin_str(y_i[j:j + 12], byteorder='big', signed=False)
    for j in range(53):
        new_y[j + 403] = mask_bin_str(y_i[j:j + 12], byteorder='little', signed=False)

    for j in range(49):
        new_y[j + 456] = mask_bin_str(y_i[j:j + 16], byteorder='big', signed=False)
    for j in range(49):
        new_y[j + 505] = mask_bin_str(y_i[j:j + 16], byteorder='little', signed=False)
    for j in range(49):
        new_y[j + 554] = mask_bin_str(y_i[j:j + 16], byteorder='big', signed=True)
    for j in range(49):
        new_y[j + 603] = mask_bin_str(y_i[j:j + 16], byteorder='little', signed=True)

    for j in range(64):
        new_y[j + 652] = int(y_i[j])

    return new_y


def generate_binary_data_from_values(values, width=8):
    values = np.rint(values)
    binary_values = [[int_to_binary_string(int(value), width)] for value in values]
    return binary_values