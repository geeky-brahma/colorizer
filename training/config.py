from dataclasses import dataclass


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str = "./runs/exp01"
    image_size: int = 256
    batch_size: int = 16
    epochs: int = 20
    lr: float = 2e-4
    num_workers: int = 2
    prefetch_factor: int = 2  # only used if num_workers > 0
    seed: int = 42
    amp: bool = False
