import tyro

from LION.pcm.config import RuntimeConfig

if __name__ == "__main__":
    cfg = tyro.cli(RuntimeConfig)
