from lightning import LightningModule, Trainer
from lightning.pytorch.cli import ArgsType, LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
import torch
import json
import src as src
import scripts as scripts
from dotenv import load_dotenv
from src.callbacks.predictionwriter import PredictionsWriter
from lightning.pytorch.callbacks import RichProgressBar

load_dotenv()
torch.set_float32_matmul_precision('medium')

class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, WandbLogger):
            config = self.parser.dump(self.config, skip_none=False, format="json")  # Required for proper reproducibility
            trainer.logger.log_hyperparams(json.loads(config))

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--paths", type=dict, default="{}")


def lightning_cli_run(args: ArgsType = None):
    cli = MyLightningCLI(save_config_callback=LoggerSaveConfigCallback, save_config_kwargs={"save_to_log_dir": False},
                        auto_configure_optimizers=False,
                        parser_kwargs={"parser_mode": "omegaconf", 
                                        "fit": {"default_config_files": ["configs/cryoten.yaml"]}},
                        args=args)

if __name__ == "__main__":
    lightning_cli_run()
