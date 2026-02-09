import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

from src.datamodule.av2_dataset import Av2Dataset, collate_fn
from src.model.trainer_forecast import Trainer as Model
from src.utils.vis import visualize_scenario


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-p", "--predict", help="", action="store_true")
    args = parser.parse_args()
    predict = args.predict

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    split = "val"
    
    # I have updated this path to match your specific server setup
    data_root = Path("/u/student/2025/cs25mtech14019/emp/data/emp")
    
    dataset = Av2Dataset(data_root=data_root, cached_split=split)

    if predict:
        chkpt_fpath = "checkpoints/empd.ckpt"
        if not os.path.exists(chkpt_fpath):
            print(f"ERROR: Checkpoint file not found at {chkpt_fpath}")
            return
            
        model = Model.load_from_checkpoint(chkpt_fpath, pretrained_weights=chkpt_fpath)
        model = model.eval().cuda()

    B = 64
    dataloader = TorchDataLoader(
        dataset,
        batch_size=B,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    # Create the folder to save images
    save_dir = "saved_images"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Images will be saved to: {os.path.abspath(save_dir)}")

    for data in tqdm(dataloader):
        if predict:
            for k in data.keys():
                if torch.is_tensor(data[k]): data[k] = data[k].cuda()
            with torch.no_grad():
                batch_pred, scores = model.predict(data, full=True)

        if "y" not in data.keys(): data["y"] = torch.zeros((data["x"].shape[0], data["x"].shape[1], 60, 2), device=data["x"].device)

        for b in range(0, data["x"].shape[0], 1):
            scene_id = data["scenario_id"][b]
            
            # This path logic matches the 'raw' folder structure we created earlier
            scene_file = data_root / ".." / split / "raw" / scene_id / ("scenario_" + scene_id + ".parquet")
            map_file = data_root / ".." / split / "raw" / scene_id / ("log_map_archive_" + scene_id + ".json")
            
            if not scene_file.exists():
                print(f"Skipping {scene_id}: Raw file not found at {scene_file}")
                continue

            scenario = scenario_serialization.load_argoverse_scenario_parquet(scene_file)
            static_map = ArgoverseStaticMap.from_json(map_file)
            
            if predict:
                prediction = batch_pred[0][b].squeeze()
                visualize_scenario(scenario, static_map, title="{}".format(scene_id), prediction=prediction, tight=True, timestep=49 if split == "test" else 50)
            else:
                visualize_scenario(scenario, static_map, title="{}".format(scene_id), tight=True, timestep=49 if split == "test" else 50)
            
            # SAVE the image to the folder
            save_path = os.path.join(save_dir, f"{scene_id}.png")
            plt.savefig(save_path)
            plt.close() # Close memory to prevent crashing

    return


if __name__ == "__main__":
    main()
