from oslactionspotting.datasets.frame import (
    ActionSpotDataset,
    ActionSpotVideoDataset,
    DaliDataSet,
    DaliDataSetVideo,
)
from .soccernet import (
    SoccerNetClips,
    SoccerNetClipsChunks,
    SoccerNetGameClips,
    SoccerNetGameClipsChunks,
)

from .json import FeatureClipChunksfromJson, FeatureClipsfromJSON
import torch
import numpy as np
import random

def build_dataset(cfg, gpu=None, default_args=None):
    """Build a dataset from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        Dataset: The constructed dataset.
    """
    if cfg.type == "SoccerNetClips" or cfg.type == "SoccerNetGames":
        if cfg.split == None:
            dataset = SoccerNetGameClips(
                path=cfg.data_root,
                features=cfg.features,
                version=cfg.version,
                framerate=cfg.framerate,
                window_size=cfg.window_size,
            )
        else:
            dataset = SoccerNetClips(
                path=cfg.data_root,
                features=cfg.features,
                split=cfg.split,
                version=cfg.version,
                framerate=cfg.framerate,
                window_size=cfg.window_size,
                train=True if cfg.type == "SoccerNetClips" else False,
            )
    elif cfg.type == "SoccerNetClipsCALF" or cfg.type == "SoccerNetClipsTestingCALF":
        if cfg.split == None:
            dataset = SoccerNetGameClipsChunks(
                path=cfg.data_root,
                features=cfg.features,
                framerate=cfg.framerate,
                chunk_size=cfg.chunk_size,
                receptive_field=cfg.receptive_field,
            )
        else:
            dataset = SoccerNetClipsChunks(
                path=cfg.data_root,
                features=cfg.features,
                split=cfg.split,
                framerate=cfg.framerate,
                chunk_size=cfg.chunk_size,
                receptive_field=cfg.receptive_field,
                chunks_per_epoch=cfg.chunks_per_epoch,
                gpu=gpu,
                train=True if cfg.type == "SoccerNetClipsCALF" else False,
            )
    elif cfg.type == "FeatureClipsfromJSON":
        dataset = FeatureClipsfromJSON(
            path=cfg.path,
            features_dir=cfg.data_root,
            classes=cfg.classes,
            framerate=cfg.framerate,
            window_size=cfg.window_size,
        )
    elif cfg.type == "FeatureVideosfromJSON":
        dataset = FeatureClipsfromJSON(
            path=cfg.path,
            features_dir=cfg.data_root,
            classes=cfg.classes,
            framerate=cfg.framerate,
            window_size=cfg.window_size,
            train=False,
        )
        # dataset = FeatureVideosfromJSON(path=cfg.path,
        #     framerate=cfg.framerate,
        #     window_size=cfg.window_size)
    elif cfg.type == "FeatureClipChunksfromJson":
        dataset = FeatureClipChunksfromJson(
            path=cfg.path,
            features_dir=cfg.data_root,
            classes=cfg.classes,
            framerate=cfg.framerate,
            chunk_size=cfg.chunk_size,
            receptive_field=cfg.receptive_field,
            chunks_per_epoch=cfg.chunks_per_epoch,
            gpu=gpu,
        )
    elif cfg.type == "FeatureVideosChunksfromJson":
        dataset = FeatureClipChunksfromJson(
            path=cfg.path,
            features_dir=cfg.data_root,
            classes=cfg.classes,
            framerate=cfg.framerate,
            chunk_size=cfg.chunk_size,
            receptive_field=cfg.receptive_field,
            chunks_per_epoch=cfg.chunks_per_epoch,
            gpu=gpu,
            train=False,
        )
    elif cfg.type == "VideoGameWithOpencv":
        dataset_len = cfg.epoch_num_frames // cfg.clip_len
        dataset = ActionSpotDataset(
            default_args["classes"],
            cfg.path,
            cfg.data_root,
            cfg.modality,
            cfg.clip_len,
            cfg.input_fps,
            cfg.extract_fps,
            dataset_len if default_args["train"] else dataset_len // 4,
            is_eval=not default_args["train"],
            crop_dim=cfg.crop_dim,
            dilate_len=cfg.dilate_len,
            mixup=cfg.mixup,
        )
    elif cfg.type == "VideoGameWithOpencvVideo":
        dataset = ActionSpotVideoDataset(
            default_args["classes"],
            cfg.path,
            cfg.data_root,
            cfg.modality,
            cfg.clip_len,
            cfg.input_fps,
            cfg.extract_fps,
            crop_dim=cfg.crop_dim,
            overlap_len=getattr(cfg, "overlap_len", cfg.clip_len // 2),
        )
    elif cfg.type == "VideoGameWithDali":
        loader_batch_size = cfg.dataloader.batch_size // default_args["acc_grad_iter"]
        dataset_len = cfg.epoch_num_frames // cfg.clip_len
        dataset = DaliDataSet(
            default_args["num_epochs"],
            loader_batch_size,
            cfg.output_map,
            (
                default_args["repartitions"][0]
                if default_args["train"]
                else default_args["repartitions"][1]
            ),
            default_args["classes"],
            cfg.path,
            cfg.modality,
            cfg.clip_len,
            dataset_len if default_args["train"] else dataset_len // 4,
            cfg.data_root,
            cfg.input_fps,
            cfg.extract_fps,
            is_eval=False if default_args["train"] else True,
            crop_dim=cfg.crop_dim,
            dilate_len=cfg.dilate_len,
            mixup=cfg.mixup,
        )
    elif cfg.type == "VideoGameWithDaliVideo":
        dataset = DaliDataSetVideo(
            cfg.dataloader.batch_size,
            cfg.output_map,
            default_args["repartitions"][1],
            default_args["classes"],
            cfg.path,
            cfg.modality,
            cfg.clip_len,
            cfg.data_root,
            cfg.input_fps,
            cfg.extract_fps,
            crop_dim=cfg.crop_dim,
            overlap_len=cfg.overlap_len,
        )
    else:
        dataset = None
    return dataset


def build_dataloader(dataset, cfg, gpu, dali, generator=None, worker_init_fn=None):
    """Build a dataloader from config dict."""
    if dali:
        return dataset
    
    # Use provided worker_init_fn or create a proper default one
    if worker_init_fn is None:
        def worker_init_fn(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
    
    dataloader_kwargs = {
        'dataset': dataset,
        'batch_size': cfg.batch_size,
        'shuffle': cfg.shuffle,
        'num_workers': cfg.num_workers if gpu >= 0 else 0,
        'pin_memory': cfg.pin_memory if gpu >= 0 else False,
        'worker_init_fn': worker_init_fn
    }
    
    # Add generator if provided (for reproducibility)
    if generator is not None:
        dataloader_kwargs['generator'] = generator
    
    if 'prefetch_factor' in cfg:
        dataloader_kwargs['prefetch_factor'] = cfg.prefetch_factor
    
    return torch.utils.data.DataLoader(**dataloader_kwargs)