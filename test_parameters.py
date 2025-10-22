from LION.utils.parameter import LIONParameter

config = {"layers": 5}

cfg = LIONParameter(**config)

cfg.is_filled()