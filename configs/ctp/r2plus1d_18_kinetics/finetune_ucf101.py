_base_ = '../r3d_18_kinetics/finetune_ucf101.py'

work_dir = './output/ctp/r2plus1d_18_kinetics/finetune_ucf101/'

model = dict(
    backbone=dict(
        type='R2Plus1D',
        pretrained='./output/ctp/r2plus1d_18_kinetics/pretraining/epoch_90.pth',
    ),
)
