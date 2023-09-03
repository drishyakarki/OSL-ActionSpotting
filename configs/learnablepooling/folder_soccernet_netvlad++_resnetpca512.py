_base_ = [
    "../_base_/datasets/folder/features_clips.py",  # dataset config
    "../_base_/models/learnablepooling.py",  # model config
]

work_dir = "outputs/learnablepooling/folder_soccernet_netvlad++_resnetpca512"

dataset = dict(
    train=dict(path="path/to/SoccerNet/train.json"),
    val=dict(path="path/to/SoccerNet/val.json"),
    test=dict(path="path/to/SoccerNet/test.json")
)

model = dict(
    neck=dict(
        type='NetVLAD++',
        vocab_size=64),
    head=dict(
        num_classes=17),
)

runner = dict(
    type="runner_JSON"
)