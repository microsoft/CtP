from ...classifiers import TSN
from ....builder import MODELS


@MODELS.register_module()
class SpeedNet(TSN):
    """ use speed info to supervised the model training.

    This idea has been mentioned in many papers:
    [1] SpeedNet: Learning the Speediness in Videos, CVPR'20
        https://arxiv.org/abs/2004.06130
    [2] Self-Supervised Spatio-Temporal Representation Learning Using Variable
        Playback Speed Prediction, arXiv 20'03
        https://arxiv.org/abs/2003.02692
    [3] Video Playback Rate Perception for Self-supervisedSpatio-Temporal
        Representation Learning, CVPR'20
        https://arxiv.org/abs/2006.11476

    Since the speed prediction can be viewed as a classification problem,
    we just need to wrapper the TSN model.
    """
    def __init__(self, *args, **kwargs):
        super(SpeedNet, self).__init__(*args, **kwargs)
