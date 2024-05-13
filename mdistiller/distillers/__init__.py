from ._base import Vanilla
from .KD import KD
from .MLKD import MLKD
from .AT import AT
from .OFD import OFD
from .RKD import RKD
from .FitNet import FitNet
from .KDSVD import KDSVD
from .CRD import CRD
from .NST import NST
from .PKT import PKT
from .SP import SP
from .Sonly import Sonly
from .VID import VID
from .ReviewKD import ReviewKD
from .DKD import DKD

distiller_dict = {
    "NONE": Vanilla,  # ........| None
    "KD": KD,  #................| Logits
    "MLKD": MLKD,  # ...........| Logits 
    "AT": AT,  # ...............| ......... Feats[1:]
    "OFD": OFD,  # .............| ..................... Preact-Feats[1:]
    "RKD": RKD,  # .............| ......................................... Pooled-Feats
    "FITNET": FitNet,  # .......| ......... Feats[hint]
    "KDSVD": KDSVD,  # .........| ......... Feats[1:]
    "CRD": CRD,  # .............| ......................................... Pooled-Feats
    "NST": NST,  # .............| ......... Feats[1:]
    "PKT": PKT,  # .............| ......................................... Pooled-Feats
    "SP": SP,  # ...............| ......... Feats[-1]
    "Sonly": Sonly,  # .........| Logits, 
    "VID": VID,  # .............| ......... Feats[1:]
    "REVIEWKD": ReviewKD,  # ...| ......... Feats[:] ...................... Pooled-Feats
    "DKD": DKD,  # .............| Logits
}
