"""
Allow users to import the BinaryHakeModel class and MVBS tensor conversion function.
"""

# TODO fix this since it doesn't work with a normal pip install but
# pip installation with editable mode seems to generate it just fine.
# from _echoregions_version import version as __version__

from hake_segmentation_inference.BinaryHakeModel import BinaryHakeModel
from hake_segmentation_inference.utils import get_MVBS_tensor

__all__ = [BinaryHakeModel, get_MVBS_tensor]
