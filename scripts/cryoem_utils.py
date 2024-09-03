import gzip
import os
import shutil
import tempfile
import traceback
import urllib
import xml.etree.ElementTree as ET
import gemmi
import mrcfile
import numpy as np
import requests
import torch
from empatches import EMPatches
from iotbx.data_manager import DataManager
from cctbx import uctbx
import iotbx.mrcfile as iotbxmrcfile
from Bio.PDB import *
from rdkit.Chem import rdchem
import math
from copy import deepcopy

emp = EMPatches()

def fetch_xml_metadata_from_emdb(emdb_id, filepath, overwrite=True):
    url = (
        "https://files.wwpdb.org/pub/emdb/structures/"
        + "EMD-"
        + emdb_id
        + "/header/"
        + "emd-"
        + emdb_id
        + ".xml"
    )

    try:
        if not overwrite:
            if os.path.exists(filepath):
                if os.stat(filepath).st_size > 0:
                    with open(filepath) as file:
                        return file.read()
                else:
                    os.remove(filepath)

        xml_string = requests.get(url).content
        with open(filepath, "wb") as file:
            file.write(xml_string)

        return xml_string

    except Exception as e:
        print("Error: Unable fetch and parse EMD Metadata XML for EMDID: " + emdb_id)
        print(e)
        if os.path.exists(filepath):
            os.remove(filepath)
        return ""

def axis_xyz_to_num(axis):
    if axis == "X":
        return 1
    if axis == "Y":
        return 2
    if axis == "Z":
        return 3

def parse_xml_metadata_of_half_maps(xml_string):
    try:
        root = ET.fromstring(xml_string)

        pdb_id = root.find("./crossreferences/pdb_list/pdb_reference/pdb_id").text
        emdb_id = root.attrib["emdb_id"]
        resolution_method = "none" # rarely missing from xml, so initialize
        for elem in root.iter():
            if elem.tag == "resolution":
                resolution = float(elem.text)
            if elem.tag == "resolution_method":
                resolution_method = elem.text

        half_maps_list = []
        half_maps_found = False
        for elem in root.iter():
            if elem.tag == "half_map":
                half_maps_found = True
                filename = elem.find("./file").text
                x_voxels = int(root.find("./dimensions/col").text)
                y_voxels = int(root.find("./dimensions/row").text)
                z_voxels = int(root.find("./dimensions/sec").text)
                x_voxel_size = float(elem.find("./pixel_spacing/x").text)
                y_voxel_size = float(elem.find("./pixel_spacing/y").text)
                z_voxel_size = float(elem.find("./pixel_spacing/z").text)
                density_min = float(elem.find("./statistics/minimum").text)
                density_max = float(elem.find("./statistics/maximum").text)
                density_avg = float(elem.find("./statistics/average").text)
                density_std = float(elem.find("./statistics/std").text)

                axis_order = []
                axis_fast = root.find("./axis_order/fast").text
                axis_medium = root.find("./axis_order/medium").text
                axis_slow = root.find("./axis_order/slow").text
                axis_order.append(axis_xyz_to_num(axis_fast))
                axis_order.append(axis_xyz_to_num(axis_medium))
                axis_order.append(axis_xyz_to_num(axis_slow))

                cell_a = float(elem.find("./cell/a").text)
                cell_b = float(elem.find("./cell/b").text)
                cell_c = float(elem.find("./cell/c").text)
                cell_alpha = float(elem.find("./cell/alpha").text)
                cell_beta = float(elem.find("./cell/beta").text)
                cell_gamma = float(elem.find("./cell/gamma").text)
                unit_cell = [cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma]
                unit_cell_ordered = [unit_cell[axis_order[0]-1], unit_cell[axis_order[1]-1], unit_cell[axis_order[2]-1], cell_alpha, cell_beta, cell_gamma]

                origin_col = int(root.find("./origin/col").text)
                origin_row = int(root.find("./origin/row").text)
                origin_sec = int(root.find("./origin/sec").text) 
                origin = (origin_col, origin_row, origin_sec)
                origin_ordered = (origin[axis_order[0]-1], origin[axis_order[1]-1], origin[axis_order[2]-1])

                dimension = [x_voxels, y_voxels, z_voxels]
                dimension_ordered = [dimension[axis_order[0]-1], dimension[axis_order[1]-1], dimension[axis_order[2]-1]]

                half_map_metadata = {
                    "filename": filename,
                    "x_voxels": x_voxels,
                    "y_voxels": y_voxels,
                    "z_voxels": z_voxels,
                    "x_voxel_size": x_voxel_size,
                    "y_voxel_size": y_voxel_size,
                    "z_voxel_size": z_voxel_size,
                    "density_min": density_min,
                    "density_max": density_max,
                    "density_avg": density_avg,
                    "density_std": density_std,
                    "axis_order": axis_order,
                    "unit_cell": unit_cell,
                    "unit_cell_ordered": unit_cell_ordered,
                    "origin": origin,
                    "origin_ordered": origin_ordered,
                    "dimension": dimension,
                    "dimension_ordered": dimension_ordered,
                }

                half_maps_list.append(half_map_metadata)

        if not half_maps_found:
            return {"error": "No Half Maps found for: " + emdb_id + ". Skipping..." + ".\n"}

        metadata = {
            "emdb_id": emdb_id,
            "pdb_id": pdb_id,
            "resolution": resolution,
            "resolution_method": resolution_method,
            "half_maps": half_maps_list,
        }

        return metadata
    except Exception as e:
        return {"error": e}

def parse_xml_metadata_of_deposited_density_map(xml_string):
    try:
        root = ET.fromstring(xml_string)

        emdb_id = root.attrib["emdb_id"]
        pdb_id = ""
        pdb_tag = root.find("./crossreferences/pdb_list/pdb_reference/pdb_id")
        if pdb_tag is not None:
            pdb_id = pdb_tag.text

        x_voxels = int(root.find("./map/dimensions/col").text)
        y_voxels = int(root.find("./map/dimensions/row").text)
        z_voxels = int(root.find("./map/dimensions/sec").text)
        x_voxel_size = float(root.find("./map/pixel_spacing/x").text)
        y_voxel_size = float(root.find("./map/pixel_spacing/y").text)
        z_voxel_size = float(root.find("./map/pixel_spacing/z").text)
        density_min = float(root.find("./map/statistics/minimum").text)
        density_max = float(root.find("./map/statistics/maximum").text)
        density_avg = float(root.find("./map/statistics/average").text)
        density_std = float(root.find("./map/statistics/std").text)
        density_recommended = float(root.find("./map/contour_list/contour/level").text)

        axis_order = []
        axis_fast = root.find("./map/axis_order/fast").text
        axis_medium = root.find("./map/axis_order/medium").text
        axis_slow = root.find("./map/axis_order/slow").text
        axis_order.append(axis_xyz_to_num(axis_fast))
        axis_order.append(axis_xyz_to_num(axis_medium))
        axis_order.append(axis_xyz_to_num(axis_slow))

        cell_a = float(root.find("./map/cell/a").text)
        cell_b = float(root.find("./map/cell/b").text)
        cell_c = float(root.find("./map/cell/c").text)
        cell_alpha = float(root.find("./map/cell/alpha").text)
        cell_beta = float(root.find("./map/cell/beta").text)
        cell_gamma = float(root.find("./map/cell/gamma").text)
        unit_cell = [cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma]
        unit_cell_ordered = [unit_cell[axis_order[0]-1], unit_cell[axis_order[1]-1], unit_cell[axis_order[2]-1], cell_alpha, cell_beta, cell_gamma]

        origin_col = int(root.find("./map/origin/col").text)
        origin_row = int(root.find("./map/origin/row").text)
        origin_sec = int(root.find("./map/origin/sec").text)
        origin = (origin_col, origin_row, origin_sec)
        origin_ordered = (origin[axis_order[0]-1], origin[axis_order[1]-1], origin[axis_order[2]-1])

        dimension = [x_voxels, y_voxels, z_voxels]
        dimension_ordered = [dimension[axis_order[0]-1], dimension[axis_order[1]-1], dimension[axis_order[2]-1]]

        resolution_method = "none"
        for elem in root.iter():
            if elem.tag == "resolution":
                resolution = float(elem.text)
            if elem.tag == "resolution_method":
                resolution_method = elem.text

        metadata = {
            "emdb_id": emdb_id,
            "pdb_id": pdb_id,
            "x_voxels": x_voxels,
            "y_voxels": y_voxels,
            "z_voxels": z_voxels,
            "x_voxel_size": x_voxel_size,
            "y_voxel_size": y_voxel_size,
            "z_voxel_size": z_voxel_size,
            "density_min": density_min,
            "density_max": density_max,
            "density_avg": density_avg,
            "density_std": density_std,
            "density_recommended": density_recommended,
            "resolution": resolution,
            "resolution_method": resolution_method,
            "axis_order": axis_order,
            "unit_cell": unit_cell,
            "unit_cell_ordered": unit_cell_ordered,
            "origin": origin,
            "origin_ordered": origin_ordered,
            "dimension": dimension,
            "dimension_ordered": dimension_ordered,
        }

        return metadata

    except Exception as e:
        print(xml_string)
        print(traceback.format_exc())
        return {"error": e}

def fetch_cryoem_deposited_map(emdb_id, file_path, overwrite=False):
    if not overwrite:
        if os.path.exists(file_path):
            if os.stat(file_path).st_size > 0:
                return True
            else:
                os.remove(file_path)

    url = (
        "https://files.wwpdb.org/pub/emdb/structures/"
        + "EMD-"
        + emdb_id
        + "/map/"
        + "emd_"
        + emdb_id
        + ".map.gz"
    )

    # Download archive
    try:
        # Store the file inside the .gz archive located at url
        with urllib.request.urlopen(url) as response:
            with gzip.GzipFile(fileobj=response) as uncompressed:
                with open(file_path, "wb") as f:
                    f.write(uncompressed.read())

        return True
    except Exception as e:
        print("Error: Downloading EMD Map ID: ", emdb_id, " failed")
        print(e)
        if os.path.exists(file_path):
            os.remove(file_path)
        return False

def fetch_cryoem_half_map(emdb_id, file_path, map_1_or_2, overwrite=False):
    if not overwrite:
        if os.path.exists(file_path):
            if os.stat(file_path).st_size > 0:
                return True
            else:
                os.remove(file_path)

    url = (
        "https://files.wwpdb.org/pub/emdb/structures/"
        + "EMD-"
        + emdb_id
        + "/other/"
        + "emd_"
        + emdb_id
        + "_half_map_"
        + str(map_1_or_2)
        + ".map.gz"
    )

    # Download archive
    try:
        # Store the file inside the .gz archive located at url
        with urllib.request.urlopen(url) as response:
            with gzip.GzipFile(fileobj=response) as uncompressed:
                with open(file_path, "wb") as f:
                    f.write(uncompressed.read())

        return True
    except Exception as e:
        print("Error: Downloading Half Map for EMD ID: ", emdb_id, " failed")
        print(e)
        if os.path.exists(file_path):
            os.remove(file_path)
        return False

def fetch_pdb_ba(pdb_id, pdb_file_path, overwrite=False):
    try:
        if not overwrite:
            if os.path.exists(pdb_file_path):
                if os.stat(pdb_file_path).st_size > 0:
                    return True
                else:
                    os.remove(pdb_file_path)

        url = "https://files.rcsb.org/download/" + pdb_id + ".pdb1.gz"

        with urllib.request.urlopen(url) as response:
            with gzip.GzipFile(fileobj=response) as uncompressed:
                file_content = uncompressed.read()
        with open(pdb_file_path, "wb") as f:
            f.write(file_content)

        return True
    except Exception as e:
        try:
            ciftmpfile = tempfile.NamedTemporaryFile(suffix=".cif")

            # Download CIF File
            url = "https://files.rcsb.org/download/" + pdb_id + "-assembly1.cif"
            urllib.request.urlretrieve(url, ciftmpfile.name)

            # Convert CIF File to PDB
            structure = gemmi.read_structure(ciftmpfile.name)
            structure.shorten_chain_names()
            structure.write_pdb(pdb_file_path)

            # os.remove(ciftmpfile.name)
            return True

        except Exception as e:
            print("Error: Failed to download PDB")
            print(e)
            os.remove(ciftmpfile.name)
            if os.path.exists(pdb_file_path):
                os.remove(pdb_file_path)
            
            try:
                url = "https://files.rcsb.org/download/" + pdb_id + ".pdb.gz"

                with urllib.request.urlopen(url) as response:
                    with gzip.GzipFile(fileobj=response) as uncompressed:
                        file_content = uncompressed.read()
                with open(pdb_file_path, "wb") as f:
                    f.write(file_content)
                
                return True
            except Exception as e:
                print("Error: Failed to download PDB")
                print(e)
                if os.path.exists(pdb_file_path):
                    os.remove(pdb_file_path)
                return False

def save_blocks_dict_as_mrc_file(
    blocks_dict, emdb_id, block_indices_file_path, metadata_file_path, mrc_file_output_path, overwrite=False
):
    try:
        if not overwrite:
            if os.path.exists(mrc_file_output_path):
                return True

        blocks_array = []

        for i in range(0,len(blocks_dict[emdb_id].keys())):
            blocks_array.append(blocks_dict[emdb_id][i])

        indices = torch.load(block_indices_file_path, map_location=torch.device('cpu'))

        merged_arr = emp.merge_patches(blocks_array, indices, mode='avg')

        xml_string = fetch_xml_metadata_from_emdb(emdb_id, metadata_file_path, overwrite=False)
        if not xml_string:
            print("Couldnt fetch metadata xml for", emdb_id)
            return False

        metadata = parse_xml_metadata_of_deposited_density_map(xml_string)
        if "error" in metadata.keys():
            print("Parsing XML Failed for: ", emdb_id)
            print(metadata["error"])
            return False

        tmpfile = tempfile.NamedTemporaryFile(suffix='.mrc').name
        with mrcfile.new(tmpfile, overwrite=True) as mrc:
            mrc.set_data(merged_arr)
            mrc.voxel_size = metadata["x_voxel_size"]
            mrc.header.mapc = metadata["axis_order"][0]
            mrc.header.mapr = metadata["axis_order"][1]
            mrc.header.maps = metadata["axis_order"][2]

        dm = DataManager()
        map_inp = dm.get_real_map(tmpfile)
        map_inp.shift_origin(desired_origin = metadata["origin_ordered"])
        iotbxmrcfile.write_ccp4_map(file_name=mrc_file_output_path,
            unit_cell=uctbx.unit_cell(metadata["unit_cell"]),
            map_data=map_inp.map_data(),
            output_axis_order=metadata["axis_order"]
        )

        return True
    except Exception as e:
        print("Failed to convert blocks into mrc for EMD", emdb_id)
        print(e)
        if os.path.exists(mrc_file_output_path): os.remove(mrc_file_output_path)
        return False

def save_array_as_mrc_file(array, mrc_file_output_path, overwrite=True):
    try:
        if not overwrite:
            if os.path.exists(mrc_file_output_path):
                return True

        with mrcfile.new(mrc_file_output_path, overwrite=True) as mrc:
            mrc.set_data(array)

        return True
    except Exception as e:
        print("Failed to convert array into mrc")
        print(e)
        if os.path.exists(mrc_file_output_path): os.remove(mrc_file_output_path)
        return False

def normalize_using_99p999_and_resample_mrc_file_and_threshold(input_mrc_file_path, output_mrc_file_path, threshold = 0, overwrite=False):
    try:
        if os.path.exists(input_mrc_file_path):
            if (os.stat(input_mrc_file_path).st_size > 0):
                pass
            else:
                print("Fail: input_mrc_file_path size seems to be zero byte.")
                return False
        else:
            print("Fail: input_mrc_file_path doesn't exist.")
            return False

        if not overwrite:
            if os.path.exists(output_mrc_file_path):
                if (os.stat(output_mrc_file_path).st_size > 0):
                    return True
                else:
                    os.remove(output_mrc_file_path)

        with mrcfile.open(input_mrc_file_path, mode="r+") as mrc:
            mapdata = mrc.data.astype(np.float32).copy()

        percentile_99p999 = np.percentile(mapdata[np.nonzero(mapdata)], 99.999)
        mapdata /= percentile_99p999
        mapdata[mapdata < threshold] = 0
        mapdata[mapdata > 1] = 1

        with mrcfile.new(output_mrc_file_path, overwrite=True) as mrc:
            mrc.set_data(mapdata)
            mrc.voxel_size = 1.0

        return True
    except Exception as e:
        print("Failed to resample_mrc_file")
        print(e)
        return False

def resample_mrc_file(input_mrc_file_path, output_mrc_file_path, overwrite=False):
    try:
        if os.path.exists(input_mrc_file_path):
            if (os.stat(input_mrc_file_path).st_size > 0):
                pass
            else:
                print("Fail: input_mrc_file_path size seems to be zero byte.")
                return False
        else:
            print("Fail: input_mrc_file_path doesn't exist.")
            return False

        if not overwrite:
            if os.path.exists(output_mrc_file_path):
                if (os.stat(output_mrc_file_path).st_size > 0):
                    return True
                else:
                    os.remove(output_mrc_file_path)

        with mrcfile.open(input_mrc_file_path, mode="r+") as mrc:
            mapdata = mrc.data.astype(np.float32).copy()

        with mrcfile.new(output_mrc_file_path, overwrite=True) as mrc:
            mrc.set_data(mapdata)
            mrc.voxel_size = 1.0

        return True
    except Exception as e:
        print("Failed to resample_mrc_file")
        print(e)
        return False

def normalize_using_99p999_only(input_mrc_file_path, output_mrc_file_path, overwrite=False):
    try:
        if os.path.exists(input_mrc_file_path):
            if (os.stat(input_mrc_file_path).st_size > 0):
                pass
            else:
                print("Fail: input_mrc_file_path size seems to be zero byte.")
                return False
        else:
            print("Fail: input_mrc_file_path doesn't exist.")
            return False

        if not overwrite:
            if os.path.exists(output_mrc_file_path):
                if (os.stat(output_mrc_file_path).st_size > 0):
                    return True
                else:
                    os.remove(output_mrc_file_path)

        with mrcfile.open(input_mrc_file_path, mode="r+") as mrc:
            mapdata = mrc.data.astype(np.float32).copy()

        percentile_99p999 = np.percentile(mapdata[np.nonzero(mapdata)], 99.999)
        mapdata /= percentile_99p999
        mapdata[mapdata < 0] = 0
        mapdata[mapdata > 1] = 1

        with mrcfile.new(output_mrc_file_path, overwrite=True) as mrc:
            mrc.set_data(mapdata)

        return True
    except Exception as e:
        print("Failed to resample_mrc_file")
        print(e)
        return False

def axis_xyz_to_num(axis):
    if axis == "X":
        return 1
    if axis == "Y":
        return 2
    if axis == "Z":
        return 3

def axis_num_to_name(axis_num):
    if axis_num == 0:
        return "x"
    elif axis_num == 1:
        return "y"
    elif axis_num == 2:
        return "z"
    else:
        print("Invalid axis number")

def ijk_to_xyz(mrc_nstart, mrc_step, point, device="cuda"):
    axes = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], device=torch.device(device))
    return ((mrc_step * axes) @ point) + (mrc_nstart * mrc_step) # origin = (mrc_nstart * mrc_step)

def get_transform(mrc_nstart, mrc_step, ijk_step, device="cuda"):
    ijk_origin = torch.tensor([0., 0., 0.], device=torch.device(device))  # HARD CODE

    xyz_o = ijk_to_xyz(mrc_nstart, mrc_step, ijk_origin, device=device)
    xyz_i = ijk_to_xyz(mrc_nstart, mrc_step, torch.tensor([ijk_origin[0] + ijk_step[0], ijk_origin[1],               ijk_origin[2]], device=torch.device(device)), device=device)
    xyz_j = ijk_to_xyz(mrc_nstart, mrc_step, torch.tensor([ijk_origin[0],               ijk_origin[1] + ijk_step[1], ijk_origin[2]], device=torch.device(device)), device=device)
    xyz_k = ijk_to_xyz(mrc_nstart, mrc_step, torch.tensor([ijk_origin[0],               ijk_origin[1],               ijk_origin[2] + ijk_step[2]], device=torch.device(device)), device=device)
    r = torch.stack([xyz_i - xyz_o, xyz_j - xyz_o, xyz_k - xyz_o])

    rinv = torch.inverse(r)
    tinv = torch.matmul(rinv, -xyz_o)
    tinv = torch.reshape(tinv, (-1, 1))
    tfinv = torch.cat([rinv, tinv], dim=1)

    return tfinv

def generate_simulated_map(emdb_id, pdb_file_path, simulated_map_file_path, metadata_file_path, device="cuda", overwrite=False):
    try:
        if not overwrite:
            if os.path.exists(simulated_map_file_path):
                if os.stat(simulated_map_file_path).st_size > 0:
                    return True
                else:
                    os.remove(simulated_map_file_path)

        xml_string = fetch_xml_metadata_from_emdb(emdb_id, metadata_file_path, overwrite=False)
        if not xml_string:
            print("metadata xml string invalid. EMD", emdb_id)
            return False

        metadata = parse_xml_metadata_of_deposited_density_map(xml_string)
        if "error" in metadata.keys():
            print("Parsing XML Failed for: ", emdb_id)
            print(metadata["error"])
            return False

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(emdb_id, pdb_file_path)

        positions = []
        atom_nums = []
        for atom in structure.get_atoms():
            atom_string = atom.get_name()[0]
            if atom_string in ["C", "N", "O", "S"]: # Only heavy atoms
                at = rdchem.Atom(atom_string)
                atom_nums.append(at.GetAtomicNum())
                positions.append(atom.get_coord())
        positions = torch.tensor(np.array(positions), device=torch.device(device))
        atom_type = torch.tensor(np.array(atom_nums), device=torch.device(device))
        # print("EMDB:", emdb_id,". Num of Atoms:", len(positions))

        axis_order = metadata["axis_order"]
        axis_order_name = [axis_num_to_name(axis_order[0]-1), axis_num_to_name(axis_order[1]-1), axis_num_to_name(axis_order[2]-1)]
        mrc_nstart = torch.tensor(metadata["origin_ordered"], device=torch.device(device))
        mrc_step = torch.tensor([metadata[axis_order_name[0]+"_voxel_size"], metadata[axis_order_name[1]+"_voxel_size"], metadata[axis_order_name[2]+"_voxel_size"]], device=torch.device(device))
        step = torch.tensor([1, 1, 1], device=torch.device(device))  # HARD CODE

        p2m_transform = get_transform(mrc_nstart, mrc_step, step, device=device)
        rot = p2m_transform[:, :3].unsqueeze(0).expand(len(positions), -1, -1)
        tra = p2m_transform[:, 3].unsqueeze(0).expand(len(positions), -1)
        tp = torch.mul(rot, positions.reshape(len(positions), 1, 3))
        ts = torch.sum(tp, dim=1)
        positions_transformed = ts + tra

        dimensions = torch.tensor(metadata["dimension_ordered"], device=torch.device(device))
        map = torch.zeros(metadata["dimension_ordered"], device=torch.device(device))

        _, indices  = emp.extract_patches(map, patchsize=20, stride=20, vox=True)

        k = torch.tensor((math.pi / (0.9 * metadata["resolution"])) ** 2, device=torch.device(device))
        theta = (k / math.pi).pow(1.5)
        
        for i in range(len(indices)):
            region = indices[i]
            start = [region[0], region[2], region[4]]
            start = torch.tensor(start, device=torch.device(device))
            end = [region[1], region[3], region[5]]
            end = torch.tensor(end, device=torch.device(device))
            
            pad_start = start - 10
            pad_end = end + 10
            pad_start[pad_start < 0] = 0
            pad_end[pad_end > dimensions] = dimensions[pad_end > dimensions]

            mask = (positions_transformed.ge(pad_start) & positions_transformed.lt(pad_end)).all(dim=1)
            pts_in_region = positions_transformed[mask]
            if len(pts_in_region) > 0:
                # print("Processing region:", i, "Num of points:", len(pts_in_region))
                grid_points = torch.stack(torch.meshgrid(
                    torch.arange(start[0], end[0], device=torch.device(device)),
                    torch.arange(start[1], end[1], device=torch.device(device)),
                    torch.arange(start[2], end[2], device=torch.device(device)), indexing="ij"),
                    dim=-1
                )
                d = (pts_in_region[:, None, None, None, :] - grid_points[None, :, :, :, :]).pow(2).sum(dim=-1).sqrt()
                val = theta * atom_type[mask][:, None, None, None] * torch.exp(-k * d.pow(2))
                map[start[0]:end[0], start[1]:end[1], start[2]:end[2]] += val.sum(dim=0)

        tmpfile = tempfile.NamedTemporaryFile(suffix='.mrc').name
        map = map.permute(2, 1, 0) # pdb follows ZYX. first convert to ZYX to XYZ, then permute to axis order

        with mrcfile.new(tmpfile, overwrite=True) as mrc:
            mrc.set_data(map.cpu().numpy())
            mrc.voxel_size = metadata["x_voxel_size"]

        dm = DataManager()
        map_inp = dm.get_real_map(tmpfile)
        map_inp.shift_origin(desired_origin = metadata["origin_ordered"])
        iotbxmrcfile.write_ccp4_map(file_name=simulated_map_file_path,
            unit_cell=uctbx.unit_cell(metadata["unit_cell"]),
            map_data=map_inp.map_data(),
            output_axis_order=metadata["axis_order"]
        )
        return True

    except KeyboardInterrupt:
        print("Interrupted: Cancel")
        if os.path.exists(simulated_map_file_path): os.remove(simulated_map_file_path)
        exit()
    
    except Exception as e:
        print("Failed to generate simulated map using pdb")
        print(e)
        return False


def prepare_dataset(emdb_id, cryoem_mrc_file_path, simulated_mrc_file_path, blocks_input_dirpath, blocks_target_dirpath, block_indices_file_path, block_size=64, stride_size=50, skip_empty=False, overwrite=False):
    input_block_filepath = None
    target_block_filepath = None
    try:
        with mrcfile.open(cryoem_mrc_file_path) as mrc:
            input_array = np.copy(mrc.data)
        
        with mrcfile.open(simulated_mrc_file_path) as mrc:
            target_array = np.copy(mrc.data)
        
        # since block_indices_file is saved only after all the blocks are saved, 
        # if this file exist, then it means all blocks are processed for this EMDB ID.
        # so we can skip it.
        if not overwrite:
            if os.path.exists(block_indices_file_path):
                if (os.stat(block_indices_file_path).st_size > 0):
                    return True
                else:
                    os.remove(block_indices_file_path)

        input_blocks, input_indices  = emp.extract_patches(input_array, patchsize=block_size, stride=stride_size, vox=True)
        target_blocks, target_indices  = emp.extract_patches(target_array, patchsize=block_size, stride=stride_size, vox=True)

        for i in range(0, len(input_blocks)):
            if skip_empty:
                if np.all(input_blocks[i] <= 0) or np.all(target_blocks[i] <= 0):
                    continue
            input_block_filepath = os.path.join(blocks_input_dirpath, emdb_id+"_"+str(i)+".mrc")
            target_block_filepath = os.path.join(blocks_target_dirpath, emdb_id+"_"+str(i)+".mrc")
            # Unexpected program exit may leave empty/corrupted files. So, we overwrite them, just to be sure.
            with mrcfile.new(input_block_filepath, overwrite=True) as mrc:
                mrc.set_data(deepcopy(input_blocks[i]))
            with mrcfile.new(target_block_filepath, overwrite=True) as mrc:
                mrc.set_data(deepcopy(target_blocks[i]))

        with open(block_indices_file_path, "wb") as f:
            torch.save(input_indices, f, pickle_protocol=5)

        return True

    except KeyboardInterrupt:
        print("Interrupted: Cancel")
        if input_block_filepath is not None:
            if os.path.exists(input_block_filepath): os.remove(input_block_filepath)
        if target_block_filepath is not None:
            if os.path.exists(target_block_filepath): os.remove(target_block_filepath)
        exit()

    except Exception as e:
        print("Failed to prepare dataset for EMD", emdb_id)
        print(e)
        if input_block_filepath is not None:
            if os.path.exists(input_block_filepath): os.remove(input_block_filepath)
        if target_block_filepath is not None:
            if os.path.exists(target_block_filepath): os.remove(target_block_filepath)
        return False

def convert_half_maps_to_full_map(
    half_map1_filepath, half_map2_filepath, full_map_filepath, overwrite=False
):
    try:
        if os.path.exists(half_map1_filepath) and os.path.exists(half_map2_filepath):
            if os.stat(half_map1_filepath).st_size > 0 and os.stat(half_map2_filepath).st_size > 0:
                pass
            else:
                print("Fail: Half maps size seems to be zero byte.")
                return False
        else:
            print("Fail: Half maps doesn't exist.")
            return False

        if not overwrite:
            if os.path.exists(full_map_filepath):
                if os.stat(full_map_filepath).st_size > 0:
                    return True
                else:
                    os.remove(full_map_filepath)

        with mrcfile.open(half_map1_filepath) as mrc:
            hm1 = mrc.data.astype(np.float32).copy()

        with mrcfile.open(half_map2_filepath) as mrc:
            hm2 = mrc.data.astype(np.float32).copy()

        full_map = 0.5 * (hm1 + hm2)

        # copy existing file but swap out the values. This way the headers stays the same.
        shutil.copyfile(half_map1_filepath, full_map_filepath)
        with mrcfile.open(full_map_filepath, mode="r+") as mrc:
            mrc.set_data(full_map)

        return True
    except Exception as e:
        print("Error: Converting half maps to full maps failed")
        print(e)
        if os.path.exists(full_map_filepath):
            os.remove(full_map_filepath)
        return False

def split_input_and_target_map_into_blocks(emdb_id, cryoem_mrc_file_path, simulated_mrc_file_path, blocks_input_dirpath, blocks_target_dirpath, block_indices_file_path, block_size=64, stride_size=50, skip_empty=False, overwrite=False):
    input_block_filepath = None
    target_block_filepath = None
    try:
        with mrcfile.open(cryoem_mrc_file_path) as mrc:
            input_array = np.copy(mrc.data)
        
        with mrcfile.open(simulated_mrc_file_path) as mrc:
            target_array = np.copy(mrc.data)
        
        # since block_indices_file is saved only after all the blocks are saved, 
        # if this file exist, then it means all blocks are processed for this EMDB ID.
        # so we can skip it.
        if not overwrite:
            if os.path.exists(block_indices_file_path):
                if (os.stat(block_indices_file_path).st_size > 0):
                    return True
                else:
                    os.remove(block_indices_file_path)

        input_blocks, input_indices  = emp.extract_patches(input_array, patchsize=block_size, stride=stride_size, vox=True)
        target_blocks, target_indices  = emp.extract_patches(target_array, patchsize=block_size, stride=stride_size, vox=True)

        for i in range(0, len(input_blocks)):
            if skip_empty:
                if np.all(input_blocks[i] <= 0) or np.all(target_blocks[i] <= 0):
                    continue
            input_block_filepath = os.path.join(blocks_input_dirpath, emdb_id+"_"+str(i)+".mrc")
            target_block_filepath = os.path.join(blocks_target_dirpath, emdb_id+"_"+str(i)+".mrc")
            # Unexpected program exit may leave empty/corrupted files. So, we overwrite them, just to be sure.
            with mrcfile.new(input_block_filepath, overwrite=True) as mrc:
                mrc.set_data(deepcopy(input_blocks[i]))
            with mrcfile.new(target_block_filepath, overwrite=True) as mrc:
                mrc.set_data(deepcopy(target_blocks[i]))

        with open(block_indices_file_path, "wb") as f:
            torch.save(input_indices, f, pickle_protocol=5)

        return True

    except KeyboardInterrupt:
        print("Interrupted: Cancel")
        if input_block_filepath is not None:
            if os.path.exists(input_block_filepath): os.remove(input_block_filepath)
        if target_block_filepath is not None:
            if os.path.exists(target_block_filepath): os.remove(target_block_filepath)
        exit()

    except Exception as e:
        print("Failed to prepare dataset for EMD", emdb_id)
        print(e)
        if input_block_filepath is not None:
            if os.path.exists(input_block_filepath): os.remove(input_block_filepath)
        if target_block_filepath is not None:
            if os.path.exists(target_block_filepath): os.remove(target_block_filepath)
        return False
