import os
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime

collection_dir = "data/collection"
csv_file = "data/testset_benchmark.csv"
output_csv_file = "data/experiments/benchmark/cryoten_benchmark_sample.csv"
output_dir = "data/experiments/benchmark/cryoten_sample"
batch_size = 40

def run_cryoten(input_map, output_map, batch_size):
    try:
        start = datetime.now()
        result = subprocess.run(
            [
                "python3",
                "eval.py",
                input_map,
                output_map,
                "--batch_size="+str(batch_size),
            ],
            check=True,
        )
        runtime = datetime.now() - start

        return runtime

    except KeyboardInterrupt:
        print("Keyboard interrupt!")
        exit()

    except Exception as e:
        print("Failed to generate map using CryoTEN")
        print(e)
        return 0

if __name__ == "__main__":
    map_list = pd.read_csv(csv_file)
    emd_list = np.copy(map_list["EMDB Map"].values)
    
    df = pd.DataFrame(columns=[
        "EMDB ID", 
        "time_taken",
    ])

    for id in emd_list:
        emdb_id = id.split("-")[1]
        map_dir = os.path.join(collection_dir, "emd_"+emdb_id)
        input_map = os.path.join(map_dir, "cryoem_deposited.mrc")
        emdb_output_dir = os.path.join(output_dir, emdb_id)
        output_map = os.path.join(emdb_output_dir, "cryoten_generated.mrc")
        os.makedirs(emdb_output_dir, exist_ok=True)
        runtime = 0

        if os.path.exists(input_map):
            print("Processing EMD", emdb_id)
            runtime = run_cryoten(input_map, output_map, batch_size)
        
        df.loc[len(df.index)] = [emdb_id, runtime]
        df.to_csv(output_csv_file, index=False)
        print("Processed EMD", emdb_id, "in", runtime)
        print("------------------------------------------------------------------")
