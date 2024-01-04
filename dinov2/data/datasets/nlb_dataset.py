import pathlib
from typing import Any, Optional, Callable, Tuple

from PIL import Image
import PIL

from dinov2.data.datasets.extended import ExtendedVisionDataset


class NLBDataset(ExtendedVisionDataset):
    def __init__(self,
                 root: str,
                 transforms: Optional[Callable] = None,
                    transform: Optional[Callable] = None,
                    target_transform: Optional[Callable] = None) -> None:

        super().__init__(root, transforms, transform, target_transform)

        self.root = pathlib.Path(root)
        
        # get all the files from direcories and subdirectories using pathlib
        files = self.root.rglob('*')

        # Filter out directories, keep only files
        files = [file for file in files if file.is_file()]

        print(len(files))

        self.images_paths = files

    def get_image_data(self, index: int) -> bytes:  # should return an image as an array
        
        image_path = self.images_paths[index]

        try:
            img = Image.open(image_path).convert(mode="RGB")
        except PIL.UnidentifiedImageError as e:
            print(f"Error opening image: {e}")

        return img
    
    def get_target(self, index: int) -> Any:
        return 0
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.images_paths)