from examples.losses.angular_loss import AngularSoftmaxLoss
from examples.losses.ce_loss import CrossEntropyLoss
from examples.losses.ge2e_loss import GE2ELoss
from examples.losses.proto_loss import ProtoLoss
from examples.losses.angular_proto import AngularProtoLoss

supported_losses = {
    "aam": AngularSoftmaxLoss,
    "ge2e": GE2ELoss,
    "cross_entropy": CrossEntropyLoss,
    "proto": ProtoLoss,
    "angular_proto": AngularProtoLoss
}