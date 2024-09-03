import os
import cryoem_utils as cu
import pandas as pd
import argparse

def collect_data(map):
    try:
        pdb_id = map["Entry ID"]
        emdb_id = map["EMDB Map"].split("-")[1]

        print("Processing EMD", emdb_id)

        map_dir = os.path.join(collection_dir, "emd_"+emdb_id)
        os.makedirs(map_dir, exist_ok=True)

        pdb_file_path = os.path.join(map_dir, pdb_id + ".pdb")
        metadata_file_path = os.path.join(map_dir, "cryoem_metadata.xml")
        cryoem_deposited_map_file_path = os.path.join(map_dir, "cryoem_deposited.mrc")

        print("===> Fetching Metadata XML and Parsing")
        xml_string = cu.fetch_xml_metadata_from_emdb(emdb_id, metadata_file_path, overwrite=False)
        if not xml_string:
            return

        print("===> Fetching CryoEM density map from EMDB")
        success = cu.fetch_cryoem_deposited_map(
            emdb_id, cryoem_deposited_map_file_path, overwrite=False
        )
        if not success:
            return
        
        print("===> Fetching respective PDB from RCSB PDB Data Bank")
        success = cu.fetch_pdb_ba(pdb_id, pdb_file_path, overwrite=False)
        if not success:
            return

        print("Success: Processed: EMD " + emdb_id + ", PDB " + pdb_id)
        print("----------------------------------------------------------------------------------------------")

    except Exception as e:
        print("Failed to process", map["EMDB Map"])
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("collect_data")
    parser.add_argument("--csv", help="csv file with Entry ID,EMDB Map,split entries", type=str, required=True)
    parser.add_argument("--collection_dir", help="path to store the data collection", type=str, default="data/collection")

    args = parser.parse_args()
    csv_file = os.path.abspath(args.csv)
    collection_dir = args.collection_dir

    map_list = pd.read_csv(csv_file)

    for i, map in map_list.iterrows():
        collect_data(map)
