"""Factory for the released MSCA-Net architecture."""
from .msca_net import MSCANet

def build_model(config):
    cfg = dict(config)
    name = cfg.pop("name", "msca_net")
    if name != "msca_net":
        raise ValueError("This release contains MSCA-Net only")
    if "kernels" in cfg:
        cfg["kernels"] = tuple(cfg["kernels"])
    return MSCANet(**cfg)
