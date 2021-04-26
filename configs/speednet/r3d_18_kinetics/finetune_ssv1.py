_base_ = ['../../recognizers/_base_/model_r3d18.py',
          '../../recognizers/_base_/runtime_ssv1.py']

work_dir = './output/speednet/r3d_18_kinetics/finetune_ssv1/'

model = dict(
    backbone=dict(
        pretrained='./output/speednet/r3d_18_kinetics/pretraining/epoch_90.pth',
    ),
    st_module=dict(temporal_size=4),
    cls_head=dict(
        num_classes=174
    )
)
