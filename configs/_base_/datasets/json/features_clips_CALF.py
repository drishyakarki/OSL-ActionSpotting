classes = ("Medical",) 

dataset = dict(
    train=dict(
        type="FeatureClipChunksfromJson",
        path="train.json",
        framerate=2,
        chunk_size=120,
        receptive_field=40,
        chunks_per_epoch=6000,
        classes=classes,
        dataloader=dict(
            num_workers=4,
            batch_size=256,
            shuffle=True,
            pin_memory=True,
        )),
    val=dict(
        type="FeatureClipChunksfromJson",
        path="val.json",
        framerate=2,
        chunk_size=120,
        receptive_field=40,
        chunks_per_epoch=6000,
        classes=classes,
        dataloader=dict(
            num_workers=4,
            batch_size=256,
            shuffle=True,
            pin_memory=True,
        )),
    test=dict(
        type="FeatureVideosChunksfromJson",
        path="test.json",
        framerate=2,
        chunk_size=120,
        receptive_field=40,
        chunks_per_epoch=6000,
        split=["test"],
        classes=classes,
        metric = "loose",
        dataloader=dict(
            num_workers=1,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
        )),
)
