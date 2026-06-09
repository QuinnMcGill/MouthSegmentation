import torch
from torch.utils.data import Dataset

class HeatmapWrapper(Dataset):
    def __init__(self, base_dataset, image_size=(256, 256), heatmap_size=64, sigma=2.0):
        self.base_dataset = base_dataset
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma

    def __len__(self):
        return len(self.base_dataset)

    def _generate_heatmaps(self, landmarks, Hm, Wm, img_h, img_w):
        k = landmarks.shape[0]
        heatmaps = torch.zeros((k, Hm, Wm), dtype=torch.float32)

        sx = Wm / img_w
        sy = Hm / img_h

        # Create a meshgrid for the heatmap of landmark coordinates
        xs = torch.arange(Wm, dtype=torch.float32)
        ys = torch.arange(Hm, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        for i in range(k):
            x, y = landmarks[i]
            cx = x * sx
            cy = y * sy
            heatmaps[i] = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * self.sigma ** 2))

        return heatmaps

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]

        image = sample["image"].float()          # [3, H, W]
        mask = sample["mask"].float()            # [2, H, W]
        item = {
            "image": image,
            "mask": mask,
            "image_path": sample["image_path"],
        }

        if "landmarks" in sample and sample["landmarks"] is not None:
            landmarks = sample["landmarks"].float()  # [19, 2]
            item["landmarks"] = landmarks

            _, H, W = image.shape
            item["heatmaps"] = self._generate_heatmaps(
                landmarks,
                Hm=self.heatmap_size,
                Wm=self.heatmap_size,
                img_h=H,
                img_w=W,
            )

        return item