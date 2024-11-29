import torch
from PIL import Image
from torch.utils.data import Dataset
import tifffile as tiff
import numpy as np

class TiffDataset(Dataset):
    def __init__(self, tiff_path, segment_size, stride, lod=7):
        self.tiff_path = tiff_path
        self.segment_size = segment_size
        self.stride = stride
        self.image = tiff.imread(tiff_path)  # Load TIFF file as a numpy array
        self.height, self.width = self.image.shape[:2]
        self.indices = self._generate_indices()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _generate_indices(self):
        """Generate top-left corner indices for all image segments."""
        indices = []
        for y in range(0, self.height - self.segment_size + 1, self.stride):
            for x in range(0, self.width - self.segment_size + 1, self.stride):
                indices.append((y, x))
        return indices

    def __len__(self):
        return len(self.indices)

    def set_lod(self, lod):
        assert(lod < 10)
        self.current_lod = lod

    def __getitem__(self, idx):
        y, x = self.indices[idx]
        segment = self.image[y:y + self.segment_size, x:x + self.segment_size]
        segment = Image.fromarray(segment)  # Convert numpy array to PIL Image
        segment = segment.convert("RGB")  # Ensure 3 channels for consistency
        segment = torch.tensor(np.array(segment), dtype=torch.float32).permute(2, 0, 1) / 255.0
        return segment.to(device)

