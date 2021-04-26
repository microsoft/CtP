_base_ = '../r3d_18_ucf101/finetune_hmdb51.py'

work_dir = './output/ctp/r2plus1d_18_ucf101/finetune_hmdb51/'

model = dict(
    backbone=dict(
        type='R2Plus1D',
        pretrained='./output/ctp/r2plus1d_18_ucf101/pretraining/epoch_300.pth',
    ),
    cls_head=dict(
        num_classes=51
    )
)
