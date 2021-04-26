dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
syncbn = True

data = dict(
    videos_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='VCOPDataset',
        data_source=dict(
            type='JsonClsDataSource',
            ann_file='ucf101/annotations/train_split_1.json',
        ),
        backend=dict(
            type='ZipBackend',
            zip_fmt='ucf101/zips/{}.zip',
            frame_fmt='img_{:05d}.jpg',
        ),
        test_mode=False,
        transform_cfg=[
                dict(type='GroupScale', scales=[(171, 128)]),
                dict(type='GroupRandomCrop', out_size=112),
                dict(
                    type='GroupToTensor',
                    switch_rgb_channels=True,
                    div255=True,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
        ],
        clip_len=16,
        min_interval=8,
        max_interval=8,
        tuple_len=3,
    )
)

# optimizer
total_epochs = 300
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[100, 200]
)
checkpoint_config = dict(interval=1, max_keep_ckpts=1, create_symlink=False)
workflow = [('train', 1)]
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ]
)
