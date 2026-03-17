import tyro

from LION.pcm.config import ExperimentConfig

if __name__ == "__main__":
    cfg = tyro.cli(ExperimentConfig)
