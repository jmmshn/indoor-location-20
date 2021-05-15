# coding: utf-8
"""
Store the wifi ssids as a set for each building
"""

import re
from monty.serialization import dumpfn, loadfn
from glob import glob
from os import path

import os

dir_path = os.path.dirname(os.path.abspath(__file__))

# Fixed variables
from tqdm.auto import tqdm

FLOOR_MAP = {
    "1F": 0,
    "2F": 1,
    "3F": 2,
    "4F": 3,
    "5F": 4,
    "6F": 5,
    "7F": 6,
    "8F": 7,
    "9F": 8,
    "B": -1,
    "B1": -1,
    "B2": -2,
    "B3": -3,
    "BF": -1,
    "F1": 0,
    "F10": 9,
    "F2": 1,
    "F3": 2,
    "F4": 3,
    "F5": 4,
    "F6": 5,
    "F7": 6,
    "F8": 7,
    "F9": 8,
    "L1": 0,
    "L10": 9,
    "L11": 10,
    "L2": 1,
    "L3": 2,
    "L4": 3,
    "L5": 4,
    "L6": 5,
    "L7": 6,
    "L8": 7,
    "L9": 8,
}

if path.exists(f"{dir_path}/building_info.json"):
    BUILDING_INFO = loadfn(f"{dir_path}/building_info.json")

if path.exists(f"{dir_path}/valid_buiding_ids.json"):
    VALID_BUILDING_IDS = loadfn(f"{dir_path}/valid_buiding_ids.json")
    BUILDING_INDEX = {v: i for i, v in enumerate(VALID_BUILDING_IDS)}


def get_building_dict(training_dir) -> dict:
    """
    Read the data structure of the taining directory and create a
    basic dictionary to hold the data
    Args:
        training_dir: the directory containing all the training data
    Returns:
        a dictionary key by the builder number
    """
    building = dict()  # key building number keys
    for ibuild, ifold in tqdm(list(enumerate(glob(training_dir + "/*")))):
        d = dict(folder=ifold, floor_dirs=dict(), wifi_ssids=set())
        d["building_id"] = ifold.split("/")[-1]
        for fl_dir in glob(d["folder"] + "/*"):
            fl = fl_dir.split("/")[-1]
            if fl not in FLOOR_MAP.keys():
                break
            d["floor_dirs"][FLOOR_MAP[fl]] = fl_dir
            d["wifi_ssids"] |= get_wifi_ids_in_folder(fl_dir)
        else:
            d["wifi_ssids"] = {v: i for i, v in enumerate(sorted(list(d["wifi_ssids"])))}
            building[d["building_id"]] = d
    return building


def get_wifi_ids_in_folder(folder: str) -> set:
    """
    read all the txt files in a folder return all the the unique wifi ssids
    Args:
        folder: the folder containing a list of txt files that represents the paths

    Returns:
        wifi ssids stored as a set

    """
    ssids = set()
    for file in glob(folder + "/*"):
        file_ssids = get_wifi_ssids_in_file(file)
        ssids |= file_ssids
    return ssids


def get_wifi_ssids_in_file(file) -> set:
    """
    Read a file and return the list of wifi ssids
    """
    with open(file, "r") as f:
        txt = f.read()
    matches = re.findall("TYPE_WIFI.*\n", txt)
    ssids = set()
    for imatch in matches:
        _, ssid, bssid, rssi, frequency, last_seen_time = imatch.split("\t")  # if match:
        ssids.add(ssid)
    return ssids


# def get_building_code(file):
#     """
#     Read a file and return the building id that shares the most wifi ssids
#     """
#     ssids_in_file = get_wifi_ssids_in_file(file)
#     vec = np.zeros(len(VALID_BUILDING_IDS)
#     for ib, db in sorted(BUILDING_INFO.items()):
#         common_ssids = set(db['wifi_ssids']) & ssids_in_file
#         ii = BUILDING_INDEX[db['building_id']]
#         vec[ii] = len(common_ssids)
#     return VALID_BUILDING_IDSnp.argmax(vec)]


if __name__ == "__main__":
    # store the building_info into a json file
    if path.exists("building_info.json") or True:
        # the basic building_info dictionary
        BUILDING_INFO = get_building_dict("/home/lik/Documents/indoor-location-data/train/")
        dumpfn(BUILDING_INFO, "building_info.json")
        # a list of valid buiding id so we have a conversion between buiding id and an index
        valid_building_ids = []
        for k, v in BUILDING_INFO.items():
            valid_building_ids.append(v["building_id"])
        valid_building_ids.sort()
        dumpfn(valid_building_ids, "valid_buiding_ids.json")
