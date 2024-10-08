import os
import cryoem_utils as cu
import pandas as pd
import argparse

def create_dataset(map):
    try:
        pdb_id = map["Entry ID"]
        emdb_id = map["EMDB Map"].split("-")[1]
        split = map["split"]
        print("Processing EMD", emdb_id)

        map_dir = os.path.join(collection_dir, "emd_"+emdb_id)
        os.makedirs(map_dir, exist_ok=True)

        pdb_file_path = os.path.join(map_dir, pdb_id + ".pdb")
        metadata_file_path = os.path.join(map_dir, "cryoem_metadata.xml")
        cryoem_deposited_map_file_path = os.path.join(map_dir, "cryoem_deposited.mrc")
        cryoem_deposited_map_processed_file_path = os.path.join(map_dir, "cryoem_deposited_processed.mrc")
        simulated_map_file_path = os.path.join(map_dir, "pdb_simulated.mrc")
        simulated_map_processed_file_path = os.path.join(map_dir, "pdb_simulated_processed.mrc")

        xml_string = cu.fetch_xml_metadata_from_emdb(emdb_id, metadata_file_path, overwrite=False)
        if not xml_string:
            return

        metadata = cu.parse_xml_metadata_of_deposited_density_map(xml_string)
        if "error" in metadata.keys():
            print("Parsing XML Failed for: ", emdb_id)
            print(metadata["error"])
            return
        
        print("===> Normalize and Resample CryoEM density map")
        success = cu.normalize_using_99p999_and_resample_mrc_file_and_threshold(
            cryoem_deposited_map_file_path, cryoem_deposited_map_processed_file_path, threshold=0, overwrite=False
        )
        if not success:
            return False
        
        # RUNNING in a GPU is much faster for larger maps and should be preffered. cpu will work fine though albeit slower.
        print("===> Generating Simulated density map from PDB")
        success = cu.generate_simulated_map(
            emdb_id, pdb_file_path, simulated_map_file_path, metadata_file_path, device="cuda", overwrite=False
        )
        if not success:
            return False

        # DONT normalize simulated map. only resample.
        print("===> Resample PDB Simulated density map")
        success = cu.resample_mrc_file(
            simulated_map_file_path, simulated_map_processed_file_path, overwrite=False
        )
        if not success:
            return False

        block_indices_dirpath = os.path.join(dataset_dir, "block_indices")
        blocks_input_dirpath = os.path.join(dataset_dir, split, "input")
        blocks_target_dirpath = os.path.join(dataset_dir, split, "target")
        os.makedirs(blocks_input_dirpath, exist_ok=True)
        os.makedirs(blocks_target_dirpath, exist_ok=True)
        os.makedirs(block_indices_dirpath, exist_ok=True)
        block_indices_file_path = os.path.join(block_indices_dirpath, "emd_"+emdb_id+".pt")

        print("===> Split processed maps into blocks and Copying to a separate dataset folder")
        if split == "train":
            # block size 64 because we random crop it into 48 size block in dataloader. Also skip empty/zero blocks in training set.
            success = cu.split_input_and_target_map_into_blocks(emdb_id, cryoem_deposited_map_processed_file_path, simulated_map_processed_file_path, 
                                        blocks_input_dirpath, blocks_target_dirpath, block_indices_file_path,
                                        block_size=64, stride_size=50, skip_empty=True, overwrite=False)
            if not success:
                return False
        else:
            # block size 48. we dont random crop it in validation and test set. 
            # Also dont skip empty/zero blocks in validation and test set, so we can reconstruct entire map for evaluation.
            success = cu.split_input_and_target_map_into_blocks(emdb_id, cryoem_deposited_map_processed_file_path, simulated_map_processed_file_path, 
                                        blocks_input_dirpath, blocks_target_dirpath, block_indices_file_path,
                                        block_size=48, stride_size=38, skip_empty=False, overwrite=False)
            if not success:
                return False

        print("Success: Processed: EMD " + emdb_id + ", PDB " + pdb_id)
        print("----------------------------------------------------------------------------------------------")

    except Exception as e:
        print("Failed to process", map["EMDB Map"])
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("create_dataset")
    parser.add_argument("--csv", help="csv file with Entry ID,EMDB Map,split entries", type=str, required=True)
    parser.add_argument("--collection_dir", help="path to data collection", type=str, default="data/collection")
    parser.add_argument("--dataset_dir", help="path to create dataset", type=str, default="data/dataset")

    args = parser.parse_args()
    csv_file = os.path.abspath(args.csv)
    dataset_dir = os.path.abspath(args.dataset_dir)
    collection_dir = os.path.abspath(args.collection_dir)

    map_list = pd.read_csv(csv_file)

    for i, map in map_list.iterrows():
        create_dataset(map)
