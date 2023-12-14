
from .soccernet import SoccerNet,SoccerNetClips,SoccerNetClipsChunks
# from .folder import FolderClips, FolderGames
from .json import FeatureClipsfromJSON, FeatureVideosfromJSON
import torch
from mmengine.config import Config, DictAction

def build_dataset(cfg, gpu,  default_args=None):
    """Build a dataset from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        Dataset: The constructed dataset.
    """
    # if cfg.type == "SoccerNetClips" or cfg.type == "SoccerNetGames":
    #     dataset = SoccerNet(path=cfg.data_root, 
    #         features=cfg.features, split=cfg.split,
    #         version=cfg.version, framerate=cfg.framerate,
    #         window_size=cfg.window_size,train=True if cfg.type == "SoccerNetClips" else False)
    if cfg.type == "SoccerNetClips" or cfg.type == "SoccerNetGames":
        dataset=SoccerNetClips(path=cfg.data_root, 
                               features=cfg.features, 
                               split=cfg.split,
                               version=cfg.version, 
                               framerate=cfg.framerate,
                               window_size=cfg.window_size,
                               train=True if cfg.type == "SoccerNetClips" else False)
    elif cfg.type == "SoccerNetClipsCALF" or cfg.type == "SoccerNetClipsTestingCALF":
        dataset=SoccerNetClipsChunks(path=cfg.data_root, features=cfg.features,
                split=cfg.split,
                framerate=cfg.framerate,
                chunk_size=cfg.chunk_size,
                receptive_field=cfg.receptive_field,
                chunks_per_epoch=cfg.chunks_per_epoch,
                gpu = gpu,
                train=True if cfg.type == "SoccerNetClipsCALF" else False)
    elif cfg.type == "FeatureClipsfromJSON":
        dataset = FeatureClipsfromJSON(path=cfg.path, 
            framerate=cfg.framerate,
            window_size=cfg.window_size)
    elif cfg.type == "FeatureVideosfromJSON":
        dataset = FeatureVideosfromJSON(path=cfg.path, 
            framerate=cfg.framerate,
            window_size=cfg.window_size)
    else:
        dataset=None
    return dataset


# def build_dataloader(dataset, batch_size, shuffle=True, max_num_worker=4, pin_memory=True):
def build_dataloader(dataset, cfg, gpu):
    """Build a dataloader from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        Dataloader: The constructed dataloader.
    """
    dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=cfg.batch_size, shuffle=cfg.shuffle,
            num_workers=cfg.num_workers if gpu >=0 else 0, 
            pin_memory=cfg.pin_memory if gpu >=0 else False)
    return dataloader
