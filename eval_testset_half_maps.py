import os
import subprocess
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("eval_testset_half_maps")
    parser.add_argument("--csv", help="csv file with Entry ID,EMDB Map,split entries", type=str, required=True)
    parser.add_argument("--collection_dir", help="path to data collection", default="data/collection", type=str)
    parser.add_argument("--output_dir", help="path to store the generated map", default="data/experiments/eval_testset_half_maps", type=str)

    args = parser.parse_args()
    csv_file = os.path.abspath(args.csv)
    collection_dir = os.path.abspath(args.collection_dir)
    output_dir = os.path.abspath(args.output_dir)

    dataset = pd.read_csv(csv_file)

    for i, map in dataset.iterrows():
        try:
            emdb_id = map["EMDB Map"].split("-")[1]
            
            map_path = os.path.join(collection_dir, "emd_"+emdb_id, "cryoem_deposited_raw_hm1.mrc")
            if not os.path.exists(map_path):
                print("Skipping EMD", emdb_id, map_path, "does not exist.")
                continue

            print("Processing EMD", emdb_id)
            emdb_output_dir = os.path.join(output_dir, "emd_"+emdb_id)
            os.makedirs(emdb_output_dir, exist_ok=True)
            output_map_path = os.path.join(emdb_output_dir, "cryoten_generated.mrc")

            result = subprocess.run(
                [
                    "python3",
                    "eval.py",
                    map_path,
                    output_map_path,
                ],
                check=True,
                # timeout=900,
                # capture_output=True,
                text=True,
            )
            print("-------------------------------------------------------------------------------------------------")
        except Exception as e:
            print("Failed to process EMD", emdb_id, e)
            print("-------------------------------------------------------------------------------------------------")
