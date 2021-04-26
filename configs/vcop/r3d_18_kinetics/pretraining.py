_base_ = '../pretraining_runtime_kinetics.py'

work_dir = './output/vcop/r3d_18_kinetics/pretraining/'

model = dict(
    type='VCOP',
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
    vcop_head=dict(
        in_channels=512,
        tuple_len=3,
        hidden_channels=512,
        dropout_ratio=0.5
    )
)
