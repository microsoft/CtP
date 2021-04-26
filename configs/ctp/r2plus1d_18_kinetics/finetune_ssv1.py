_base_ = '../r3d_18_kinetics/finetune_ssv1.py'

work_dir = './output/ctp/r2plus1d_18_kinetics/finetune_ssv1/'

model = dict(
    backbone=dict(
        type='R2Plus1D',
        pretrained='./output/ctp/r2plus1d_18_kinetics/pretraining/epoch_90.pth',
    ),
    st_module=dict(temporal_size=4),
    cls_head=dict(
        num_classes=174
    )
)
