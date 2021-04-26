_base_ = '../pretraining_runtime_kinetics.py'

work_dir = './output/memdpc/r3d_18_kinetics/pretraining/'

model = dict(
    type='MemDPC_BD',
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
        down_sampling_temporal=[False, False, True, True],
        channel_multiplier=1.0,
        bottleneck_multiplier=1.0,
        with_bn=True,
        pretrained=None,
    ),
    feature_size=256,
    hidden_size=512,
    sample_size=112,
    num_seq=8,
    seq_len=5,
    pred_step=3,
    mem_size=1024,
    spatial_downsample_stride=16,
)

