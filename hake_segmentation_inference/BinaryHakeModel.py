# Import packages
import torch
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50


class BinaryHakeModel(torch.nn.Module):
    """
    Binary class for Hake segmentation models using DeepLabV3 architecture.

    Methods
    -------
    forward(input_tensor: torch.Tensor) -> dict[torch.Tensor]:
        Forward pass of the model, returning the backbone, classifier, interpolated, and softmax output.
    """

    def __init__(self) -> None:
        super(BinaryHakeModel, self).__init__()
        self.network = deeplabv3_resnet50(num_classes=2)

    def forward(
        self, input_tensor: torch.Tensor, softmax_temperature: float = 1.0
    ) -> dict[torch.Tensor]:
        """
        Forward pass of the model, returning the the backbone, classifier, interpolated, and softmax output.

        Parameters
        ----------
        input_tensor : torch.Tensor
            MVBS tensor.
        softmax_temperature : float
            Temperature scaling factor applied before softmax. Default is 1.0.

        Returns
        -------
        dict[torch.Tensor]
            Backbone, classifier, interpolated, and softmax output.
        """
        # Grab tensor shape
        input_shape = input_tensor.shape[-2:]

        # Forward through the backbone and capture intermediate output
        backbone_output = self.network.backbone(input_tensor)["out"]

        # Forward through classifier
        classifier_output = self.network.classifier(backbone_output)
        interpolated_output = F.interpolate(
            classifier_output, size=input_shape, mode="bilinear", align_corners=True
        )

        # Calculate softmax output
        softmax_output = F.softmax(interpolated_output / softmax_temperature, dim=1)

        return {
            "backbone_output": backbone_output,
            "classifier_output": classifier_output,
            "interpolated_output": interpolated_output,
            "softmax_output": softmax_output,
        }
