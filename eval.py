import subprocess
import tempfile
from main import lightning_cli_run
import argparse
import mrcfile
import empatches
import os
import scripts.cryoem_utils as cu
import numpy as np
from copy import deepcopy
import torch
import glob
import shutil
from iotbx.data_manager import DataManager
from cctbx import uctbx
import iotbx.mrcfile as iotbxmrcfile

emp = empatches.EMPatches()

def prepare_input(input_map_file_path, temp_input_dir):
    with mrcfile.open(input_map_file_path) as mrc:
        mapdata = mrc.data.astype(np.float32).copy()
    
    # Normalize the input
    percentile_99p999 = np.percentile(mapdata[np.nonzero(mapdata)], 99.999)
    mapdata /= percentile_99p999
    mapdata[mapdata < 0] = 0
    mapdata[mapdata > 1] = 1

    blocks, indices = emp.extract_patches(mapdata, patchsize=48, stride=38, vox=True)

    for i in range(0, len(blocks)):
        input_block_filepath = os.path.join(temp_input_dir, str(i)+".mrc")

        with mrcfile.new(input_block_filepath, overwrite=True) as mrc:
            mrc.set_data(deepcopy(blocks[i]))

    block_indices_file_path = os.path.join(temp_input_dir, "block_indices.pt")
    torch.save(indices, block_indices_file_path, pickle_protocol=5)

def merge_output(temp_output_dir, output_map_file_path, temp_input_dir, input_map_file_path):
    prediction_output_files = glob.glob(os.path.join(temp_output_dir, "predictions_output_*.pt"))
    predictions_filename_files = glob.glob(os.path.join(temp_output_dir, "predictions_filename_*.pt"))
    block_indices_ref = torch.load(os.path.join(temp_input_dir, "block_indices.pt"))

    if len(prediction_output_files) != len(predictions_filename_files):
        print("Output files and filename files don't match")
        return False

    preds = [None]*len(block_indices_ref)
    indices = [None]*len(block_indices_ref)

    for i in range(len(prediction_output_files)):
        predictions_partial = torch.load(os.path.join(temp_output_dir, "predictions_output_"+str(i)+".pt"))
        filename_partial = torch.load(os.path.join(temp_output_dir, "predictions_filename_"+str(i)+".pt"))

        if len(predictions_partial) != len(filename_partial):
            print("number of predictions and indices doesnt match")
            return False

        for i in range(len(predictions_partial)):
            cube_index = int(filename_partial[i].split("/")[-1].split(".")[0])
            preds[cube_index] = predictions_partial[i]
            indices[cube_index] = block_indices_ref[cube_index]

    merged_arr = emp.merge_patches(preds, indices, mode="avg")

    # WORKS FOR ALL MAPS EXCEPT EMD-22845. BELOW APPROACH FIXES IT
    # ------------------------------------------------------------
    # shutil.copyfile(input_map_file_path, output_map_file_path)
    # with mrcfile.open(output_map_file_path, mode='r+') as mrc:
    #     mrc.set_data(merged_arr)
    
    # NEW APPROACH. YET TO CONFIRM IT WORKS FOR ALL MAPS. BUT FIXES EMD-22845
    # -----------------------------------------------------------------------
    with mrcfile.open(input_map_file_path, mode='r+') as mrc:
        voxel_size = np.array(mrc.voxel_size.data)
        axis_order = [mrc.header.mapc, mrc.header.mapr, mrc.header.maps]
        origin = np.array(mrc.nstart.data)
        unit_cell = np.array(mrc.header.cella)
        
    # unit_cell = [merged_arr.shape[0] * voxel_size['x'], merged_arr.shape[1] * voxel_size['y'], merged_arr.shape[2] * voxel_size['z']]
    tmpfile = tempfile.NamedTemporaryFile(suffix='.mrc').name
    with mrcfile.new(tmpfile, overwrite=True) as mrc:
        mrc.set_data(merged_arr)
        mrc.voxel_size = voxel_size
        mrc.header.mapc = axis_order[0]
        mrc.header.mapr = axis_order[1]
        mrc.header.maps = axis_order[2]
    
    dm = DataManager()
    map_inp = dm.get_real_map(tmpfile)
    map_inp.shift_origin(desired_origin = [int(origin['x']), int(origin['y']), int(origin['z'])])
    iotbxmrcfile.write_ccp4_map(file_name=output_map_file_path,
        unit_cell=uctbx.unit_cell([float(unit_cell['x']), float(unit_cell['y']), float(unit_cell['z'])]),
        map_data=map_inp.map_data(),
        output_axis_order=axis_order
    )
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser("eval.py")
    parser.add_argument("input_map_path", help="Input MRC map file path", type=str)
    parser.add_argument("output_map_path", help="Output map file path", type=str)
    parser.add_argument("--batch_size", help="batch size", type=int, default=90)
    parser.add_argument("--config", help="config file path", type=str, default="configs/cryoten.yaml")
    parser.add_argument("--ckpt_path", help="ckpt_path file path", type=str, default="cryoten.ckpt")

    args = parser.parse_args()
    input_map_file_path = os.path.abspath(args.input_map_path)
    output_map_file_path = os.path.abspath(args.output_map_path)
    ckpt_path = os.path.abspath(args.ckpt_path)
    batch_size = args.batch_size
    config = args.config

    if not os.path.exists(input_map_file_path):
        print(f"Error: Given map path does not exist: {input_map_file_path}")
        exit()

    if os.path.exists(output_map_file_path):
        print("Skip. Output map already exist: "+output_map_file_path)
        exit()

    temp_dir = tempfile.mkdtemp()
    temp_input_dir = os.path.join(temp_dir, "input")
    temp_output_dir = os.path.join(temp_dir, "output")
    os.makedirs(temp_input_dir, exist_ok=True)
    os.makedirs(temp_output_dir, exist_ok=True)

    try:
        print("==> Spliting map into blocks")
        prepare_input(input_map_file_path, temp_input_dir)

        print("==> Generate output blocks using CryoSR")
        result = subprocess.run(
            [
                "python3",
                "main.py",
                "predict",
                "--config="+config,
                "--data.batch_size="+str(batch_size),
                "--data.num_workers=8",
                "--ckpt_path="+ckpt_path,
                "--data.predict_dataset_dir="+temp_input_dir,
                "--trainer.callbacks+=PredictionsWriter",
                "--trainer.callbacks.output_dir="+temp_output_dir,
                "--trainer.callbacks.write_interval=epoch",
                "--trainer.logger=False",
                # "--trainer.devices=1",
            ],
            check=True,
            # timeout=900,
            # capture_output=True,
            text=True,
        )

        print("==> Reconstructing output density map from generated blocks")
        success = merge_output(temp_output_dir, output_map_file_path, temp_input_dir, input_map_file_path)
        if success:
            print(f"==> Successfully generated map: {output_map_file_path}")
        else:
            print(f"==> Failed to generated map")
    except Exception as e:
        print("Error: "+str(e))
    finally:
        print("==> Deleting temporary files: "+temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
