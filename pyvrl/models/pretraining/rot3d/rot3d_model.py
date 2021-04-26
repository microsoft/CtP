from ...classifiers import TSN
from ....builder import MODELS


@MODELS.register_module()
class Rot3D(TSN):

    """ The rotation prediction is a classification problem.
    For simplicity, we fully inherit the recognizer object to classify
    the rotation degree (0, 90, 270, 360)

    [1] Self-Supervised Spatiotemporal Feature Learning via Video Rotation
        Prediction
        Longlong Jing, Xiaodong Yang, Jingen Liu, Yingli Tian
        https://arxiv.org/abs/1811.11387
    """

    def __init__(self, *args, **kwargs):
        super(Rot3D, self).__init__(*args, **kwargs)
        assert self.cls_head.num_classes == 4, \
            "The number of classes should be 4 in Rot3D pretraining task."

