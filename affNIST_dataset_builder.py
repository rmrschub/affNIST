import urllib.parse
from typing import Any

import tensorflow_datasets as tfds
import requests
import io
import numpy as np
from tensorflow_datasets.core import split_builder as split_builder_lib

# affNIST constants
_affNIST_URL = "https://www.cs.toronto.edu/~tijmen/affNIST/"
_affNIST_DOWNLOAD_URL = "https://www.cs.toronto.edu/~tijmen/affNIST/"
_affNIST_TRAIN_DATA_FILENAME = "affNIST_training_x.npz"
_affNIST_TRAIN_LABELS_FILENAME = "affNIST_training_y.npz"
_affNIST_VALIDATION_DATA_FILENAME = "affNIST_validation_x.npz"
_affNIST_VALIDATION_LABELS_FILENAME = "affNIST_validation_y.npz"
_affNIST_TEST_DATA_FILENAME = "affNIST_test_x.npz"
_affNIST_TEST_LABELS_FILENAME = "affNIST_test_y.npz"

_affNIST_IMAGE_SIZE = 40
_affNIST_IMAGE_SHAPE = (_affNIST_IMAGE_SIZE, _affNIST_IMAGE_SIZE, 1)
_affNIST_NUM_CLASSES = 10
_TRAIN_EXAMPLES = 1600000
_VALIDATION_EXAMPLES = 320000
_TEST_EXAMPLES = 320000

_affNIST_CITATION = """
@online{affNIST,
  author={Tijmen Tieleman},
  title={The {affNIST} dataset for machine learning},
  year={2013},
  url={https://www.cs.toronto.edu/~tijmen/affNIST/},
  urldate={2023-06-01}
}
"""


class AffNIST(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for affNIST dataset."""

    URL = _affNIST_DOWNLOAD_URL
    VERSION = tfds.core.Version('1.0.0')

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(affNIST): Specifies the tfds.core.DatasetInfo object
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(shape=_affNIST_IMAGE_SHAPE),
                "label": tfds.features.ClassLabel(num_classes=_affNIST_NUM_CLASSES),
            }),
            supervised_keys=('image', 'label'),  # Set to `None` to disable
            homepage=_affNIST_URL,
            citation=_affNIST_CITATION,
        )

    def _split_generators(self):
        """Returns SplitGenerators."""
        filenames = {
            "train_data": _affNIST_TRAIN_DATA_FILENAME,
            "train_labels": _affNIST_TRAIN_LABELS_FILENAME,
            "validation_data": _affNIST_VALIDATION_DATA_FILENAME,
            "validation_labels": _affNIST_VALIDATION_LABELS_FILENAME,
            "test_data": _affNIST_TEST_DATA_FILENAME,
            "test_labels": _affNIST_TEST_LABELS_FILENAME,
        }

        data = {
            k: np.load(io.BytesIO(requests.get(urllib.parse.urljoin(self.URL, v)).content))
            for k, v in filenames.items()
        }

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs=dict(
                    num_examples=_TRAIN_EXAMPLES,
                    data_path=data["train_data"],
                    label_path=data["train_labels"],
                ),
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs=dict(
                    num_examples=_VALIDATION_EXAMPLES,
                    data_path=data["validation_data"],
                    label_path=data["validation_labels"],
                ),
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs=dict(
                    num_examples=_TEST_EXAMPLES,
                    data_path=data["test_data"],
                    label_path=data["test_labels"],
                ),
            ),
        ]

    def _generate_examples(self, **kwargs: Any) -> split_builder_lib.SplitGenerator:
        pass
