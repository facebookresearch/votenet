from pathlib import Path
from torch.utils.data import IterableDataset


class SceneDataset(IterableDataset):
    """
    A dataset representing an infinite stream of noise images of specified dimensions.
    """

    def __init__(self, path: Path):
        """
        :param num_classes: Number of classes (labels)
        :param C: Number of channels per image
        """
        super().__init__()
        self.path = path
        self.image_dim = (C, W, H)

    def __iter__(self) -> Iterator[Tuple[Tensor, int]]:
        """
        :return: An iterator providing an infinite stream of random labelled images.
        """
        while True:
            X, y = random_labelled_image(self.image_dim, self.num_classes)
            yield X, y