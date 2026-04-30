from PIL import Image
from random import randint
from torchvision.transforms import ToTensor, ToPILImage
from typing import List, Literal, Tuple, Union

import torch
import torchvision.transforms.functional as F

class DataAugmentation:
	"""
	Base class for Data Augmentation.
	"""
	def __init__(self, box_format: Literal["xyxy", "xywh", "xcycwh"], **kwargs):
		"""
		Initializes the DataAugmentation class.

		:param List[Callable] **transformations**: Transformations to apply to the images.
		:param ****kwargs**: Keywords arguments.
		"""
		keys = ["brightness", "contrast", "saturation"]
		for key in kwargs.keys():
			if key in keys:
				keys.pop(keys.index(key))

		assert len(keys) == 0, f"Some keys for data augmentation are not present: {keys}."
		self._transforms = [self._brightness, self._contrast, self._horizontal_flip, self._saturation]
		self._to_tensor = ToTensor()
		self._box_format = box_format
		self._kwargs = kwargs		

	def __call__(self, image: Image.Image, bboxes: List[List[Union[int, float]]]) -> Tuple[Image.Image, List[List[Union[int, float]]]]:
		"""
		Built-in Python method that allows to call an instance like function.

		:param Image **image**: Image to transform.
		:param List[List[Union[int, float]]] **bboxes**
		:return: Image and bboxes transformed.
		:rtype: Tuple[Image.Image, List[List[Union[int, float]]]]
		"""
		image = self._to_tensor(image)
		bboxes = torch.tensor(bboxes)
		self._has_bboxes = len(bboxes) > 0
		for transform in self._transforms:
			image, bboxes = transform(image, bboxes)
		return image, bboxes.tolist()

	def _horizontal_flip(self, image: Image.Image, bboxes: List[List[float]]) -> Tuple[Image.Image, List[List[float]]]:
		"""
		Method that flips horizontaly images.

		:param Image **image**: Image to modify.
		:param List[List[float]] **bboxes**: Bounding boxes contains in the image.
		:return: Image verticaly flip and its bounding boxes.
		:rtype: Tuple[Image.Image, List[List[float]]]
		"""
		new_image = F.hflip(image)
		if self._has_bboxes:
			if self._box_format == "xyxy":
				bboxes_width = bboxes[..., 3] - bboxes[..., 1]
				bboxes[..., 1] = 1 - bboxes[..., 3]
				bboxes[..., 3] = bboxes[..., 1] + bboxes_width
			elif self._box_format == "xywh":
				bboxes[..., 1] = 1 - (bboxes[..., 1] + bboxes[..., 3])
			elif self._box_format == "xcycwh":
				bboxes[..., 1] = 1 - bboxes[..., 1]
		return new_image, bboxes

	def _vertical_flip(self, image: Image.Image, bboxes: torch.Tensor) -> Tuple[Image.Image, List[List[float]]]:
		"""
		Method that flips verticaly images.

		:param Image **image**: Image to modify.
		:param List[List[float]] **bboxes**: Bounding boxes contains in the image.
		:return: Image verticaly flip and its bounding boxes.
		:rtype: Tuple[Image.Image, List[List[float]]]
		"""
		new_image = F.vflip(image)
		if self._has_bboxes:
			if self._box_format == "xyxy":
				bboxes_height = bboxes[..., 4] - bboxes[..., 2]
				bboxes[..., 2] = 1 - bboxes[..., 4]
				bboxes[..., 4] = bboxes[..., 2] + bboxes_height
			elif self._box_format == "xywh":
				bboxes[..., 2] = 1 - (bboxes[..., 2] + bboxes[..., 4])
			elif self._box_format == "xcycwh":
				bboxes[..., 2] = 1 - bboxes[..., 2]
		return new_image, bboxes

	def _brightness(self, image: Image.Image, bboxes: List[List[float]]) -> Tuple[Image.Image, List[List[float]]]:
		"""
		Method that applies a britghness factor.

		:param Image **image**: Image to modify.
		:param List[List[float]] **bboxes**: Bounding boxes contains in the image.
		:return: Image modified.
		:rtype: Tuple[Image.Image, List[List[float]]]
		"""
		new_image = F.adjust_brightness(image, self._kwargs["brightness"])
		return new_image, bboxes

	def _contrast(self, image: Image.Image, bboxes: List[List[float]]) -> Tuple[Image.Image, List[List[float]]]:
		"""
		Method that applies a contrast factor.

		:param Image **image**: Image to modify.
		:param List[List[float]] **bboxes**: Bounding boxes contains in the image.
		:return: Image modified.
		:rtype: Tuple[Image.Image, List[List[float]]]
		"""
		new_image = F.adjust_contrast(image, self._kwargs["contrast"])
		return new_image, bboxes

	def _saturation(self, image: Image.Image, bboxes: List[List[float]]) -> Tuple[Image.Image, List[List[float]]]:
		"""
		Method that applies a saturation factor.

		:param Image **image**: Image to modify.
		:param List[List[float]] **bboxes**: Bounding boxes contains in the image.
		:return: Image modified.
		:rtype: Tuple[Image.Image, List[List[float]]]
		"""
		new_image = F.adjust_saturation(image, self._kwargs["saturation"])
		return new_image, bboxes

	def _ligthning_noise(self, image: Image.Image, bboxes: List[List[float]]) -> Tuple[Image.Image, List[List[float]]]:
		"""
		Method that swaps image channels colors.

		:param Image **image**: Image to modify.
		:param List[List[float]] **bboxes**: Bounding boxes contains in the image.
		:return: Image modified.
		:rtype: Tuple[Image.Image, List[List[float]]]
		"""
		new_image = image
		perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2),
				(1, 2, 0), (2, 0, 1), (2, 1, 0))
		swap = perms[randint(0, len(perms)- 1)]
		new_image = new_image[swap, :, :]
		return new_image, bboxes