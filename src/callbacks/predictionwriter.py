from lightning.pytorch.callbacks import BasePredictionWriter
import os
import torch

class PredictionsWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        os.makedirs(self.output_dir, exist_ok=True)
        output_file = os.path.join(self.output_dir, f"predictions_output_{trainer.global_rank}.pt")
        block_file_name = os.path.join(self.output_dir, f"predictions_filename_{trainer.global_rank}.pt")

        array = []
        block_names = []
        for batch in predictions:
            blocks, files = batch
            for i in range(len(blocks)):
                output = blocks[i]
                file = files[i]
                array.append(output[0].cpu().numpy())
                block_names.append(file)
            
        torch.save(array, output_file)
        torch.save(block_names, block_file_name)
