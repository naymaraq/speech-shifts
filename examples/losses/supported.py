from examples.losses.angular_loss import AngularSoftmaxLoss
from examples.losses.ce_loss import CrossEntropyLoss
from examples.losses.ge2e_loss import GE2E
from examples.losses.proto_loss import ProtoLoss

supported = {
    "aam": AngularSoftmaxLoss,
    "ge2e": GE2E,
    "cross_entropy": CrossEntropyLoss,
    "proto": ProtoLoss
}