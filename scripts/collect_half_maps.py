import os
import shutil
import cryoem_utils as cu
import pandas as pd
import argparse

def collect_half_maps(map, collection_dir):
    try:
        emdb_id = map["EMDB Map"].split("-")[1]
        print("Processing EMD " + emdb_id)

        map_dir = os.path.join(collection_dir, "emd_"+emdb_id)
        os.makedirs(map_dir, exist_ok=True)

        cryoem_deposited_map_raw_hm1_file_path = os.path.join(
            map_dir, "cryoem_deposited_raw_hm1.mrc"
        )
        cryoem_deposited_map_raw_hm2_file_path = os.path.join(
            map_dir, "cryoem_deposited_raw_hm2.mrc"
        )
        cryoem_deposited_map_raw_fm_file_path = os.path.join(
            map_dir, "cryoem_deposited_raw_fm.mrc"
        )

        print("===> Fetching CryoEM Half map 1 from EMDB")
        success = cu.fetch_cryoem_half_map(
            emdb_id, cryoem_deposited_map_raw_hm1_file_path, 1, overwrite=False
        )
        if not success:
            raise Exception('hm1')
        
        print("===> Fetching CryoEM Half map 2 from EMDB")
        success = cu.fetch_cryoem_half_map(
            emdb_id, cryoem_deposited_map_raw_hm2_file_path, 2, overwrite=False
        )
        if not success:
            raise Exception('hm2')
        
        print("===> Combining CryoEM Half maps to form raw full map")
        success = cu.convert_half_maps_to_full_map(
            cryoem_deposited_map_raw_hm1_file_path, cryoem_deposited_map_raw_hm2_file_path, cryoem_deposited_map_raw_fm_file_path, overwrite=False
        )
        if not success:
            raise Exception('fm')

        print("Success: Fetched half maps for EMD " + emdb_id)
        print("----------------------------------------------------------------------------------------------")
        return True

    except Exception as e:
        print("Failed to fetch half maps for EMD", map["EMDB Map"], e)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser("collect_half_maps")
    parser.add_argument("--csv", help="csv file with Entry ID,EMDB Map,split entries", type=str, required=True)
    parser.add_argument("--collection_dir", help="path to the collection directory", type=str, default="data/collection")

    args = parser.parse_args()
    csv_file = os.path.abspath(args.csv)
    collection_dir = args.collection_dir

    map_list = pd.read_csv(csv_file)
    for i, map in map_list.iterrows():
        collect_half_maps(map, collection_dir)
