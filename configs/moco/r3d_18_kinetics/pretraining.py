_base_ = '../pretraining_runtime_kinetics.py'

work_dir = './output/moco/r3d_18_kinetics/pretraining/'

model = dict(
    type='MoCo',
    backbone=dict(
        type='R3D',
        depth=18,
        num_stages=4,
        stem=dict(
            temporal_kernel_size=3,
            temporal_stride=1,
            in_channels=3,
            with_pool=False,
        ),
        down_sampling=[False, True, True, True],
        channel_multiplier=1.0,
        bottleneck_multiplier=1.0,
        with_bn=True,
        pretrained=None,
    ),
    in_channels=512,
    out_channels=128,
    queue_size=65536,
    momentum=0.999,
    temperature=0.20,
    mlp=True
)

