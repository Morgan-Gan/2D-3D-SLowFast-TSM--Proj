
import backbone.base
from backbone.slowfastnet import tsm as tsmmodel
from backbone.hidden_for_roi_maxpool import hidden50
class tsm(backbone.base.Base):

    def __init__(self):
        super().__init__(False)

    def features(self):
        print("TSM")
        tsm = tsmmodel()
        hidden = hidden50()
        num_features_out = 2048    ## 2304
        num_hidden_out = 2048*3*3  ## 2304*3*3

        return tsm, hidden, num_features_out, num_hidden_out

if __name__ == '__main__':
    s=tsm()
    s.features()
