"""Networks: MONAI ResNet18 with one output (the log-risk)."""
import monai


def build_model(cfg, n_slices=25):
    if cfg.dims == 2:   # the n_slices axial slices of the ROI are the input channels
        return monai.networks.nets.resnet18(spatial_dims=2, n_input_channels=n_slices, num_classes=1,
                                            conv1_t_stride=2)
    return monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=1, num_classes=1, conv1_t_stride=2)
