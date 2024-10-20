import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class T:

    @staticmethod
    def retrieve_transforms(h: dict) -> callable:
        transform = h['redundancy_method']
        if transform == 'random_transform':
            trans_func: callable = T.subsample_with_random_transform
        else:
            raise ValueError(f"Unknown transform: {transform}")  # TODO: add more transforms
        return trans_func

    @staticmethod
    def subsample(x: torch.Tensor, downscale_factor=2, noise_factor=0.05):
        assert x.dim() == 3  # (C, H, W)

        # Downsample using average pooling
        x = F.avg_pool2d(x, kernel_size=downscale_factor)

        # Introduce small random noise
        noise = torch.randn_like(x) * noise_factor
        x = x + noise

        # Clip values to stay within the valid range (e.g., for MNIST this is [0, 1])
        x = torch.clamp(x, 0.0, 1.0)

        return x

    @staticmethod
    def subsample_with_random_stride(x: torch.Tensor, downscale_factor=2, max_offset=1):
        assert x.dim() == 3  # (C, H, W)

        # Randomly shift the starting point of the pooling window
        offset_h = torch.randint(0, max_offset + 1, (1,)).item()
        offset_w = torch.randint(0, max_offset + 1, (1,)).item()

        # Apply padding to ensure pooling still works after shifting
        x = F.pad(x, (offset_w, 0, offset_h, 0), mode='reflect')

        # Downsample using average pooling
        x = F.avg_pool2d(x, kernel_size=downscale_factor, stride=downscale_factor)

        return x

    @staticmethod
    def subsample_with_patch_dropout(x: torch.Tensor, patch_size=2, dropout_prob=0.2):
        assert x.dim() == 3  # (C, H, W)

        # Divide the image into patches and randomly drop some
        c, h, w = x.size()
        patches_per_dim = h // patch_size
        mask = torch.rand(patches_per_dim, patches_per_dim) > dropout_prob

        # Reshape mask to match image resolution
        mask = mask.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
        mask = mask.unsqueeze(0).expand(c, -1, -1)  # expand to channel dimension

        # Apply the mask (dropout)
        x = x * mask

        # Downsample after patch dropout
        x = F.avg_pool2d(x, kernel_size=patch_size)

        return x

    @staticmethod
    def subsample_with_random_transform(x: torch.Tensor, downscale_factor=2):
        assert x.dim() == 3  # (C, H, W)

        # Random affine transformations (rotation, translation, scaling)
        angle = torch.randn(1).item() * 45  # Random rotation (-10 to 10 degrees)
        translate = [torch.randint(-2, 3, (1,)).item(), torch.randint(-2, 3, (1,)).item()]
        scale = 1.0 + torch.randn(1).item() * 0.1  # Scale within [0.9, 1.1]

        # Apply transformations (converted to PIL Image for easy manipulation)
        x_pil = TF.to_pil_image(x.squeeze(0))  # Remove channel dim for MNIST (assumed grayscale)
        x_pil = TF.affine(x_pil, angle=angle, translate=translate, scale=scale, shear=0)

        # Convert back to tensor and add channel dim again
        x = TF.to_tensor(x_pil).unsqueeze(0)

        # Downsample using average pooling
        x = F.avg_pool2d(x, kernel_size=downscale_factor)

        # Remove channel dimension for imshow compatibility
        x = x.squeeze(0)

        return x

    @staticmethod
    def subsample_with_pixel_swapping(x: torch.Tensor, swap_prob=0.1, downscale_factor=2):
        assert x.dim() == 3  # (C, H, W)

        # Generate a mask for swapping pixels
        c, h, w = x.size()
        swap_mask = torch.rand(h, w) < swap_prob

        # Add channel dimension to the mask
        swap_mask = swap_mask.unsqueeze(0).expand(c, -1, -1)

        # Shift the image by one pixel in a random direction and swap pixels according to the mask
        shift_direction = torch.randint(0, 4, (1,)).item()  # Randomly pick direction: 0=up, 1=down, 2=left, 3=right
        if shift_direction == 0:
            x_shifted = F.pad(x[:, 1:, :], (0, 0, 0, 1), mode='reflect')
        elif shift_direction == 1:
            x_shifted = F.pad(x[:, :-1, :], (0, 0, 1, 0), mode='reflect')
        elif shift_direction == 2:
            x_shifted = F.pad(x[:, :, 1:], (0, 1, 0, 0), mode='reflect')
        else:
            x_shifted = F.pad(x[:, :, :-1], (1, 0, 0, 0), mode='reflect')

        x[swap_mask] = x_shifted[swap_mask]

        # Downsample after swapping
        x = F.avg_pool2d(x, kernel_size=downscale_factor)

        return x

    @staticmethod
    def rnd_subsample_method(x: torch.Tensor):
        return T.subsample_with_random_transform(x)
        # methods = [subsample,
        #            subsample_with_random_stride,
        #            subsample_with_patch_dropout,
        #            subsample_with_random_transform,
        #            subsample_with_pixel_swapping
        #            ]
        # return methods[torch.randint(0, len(methods), (1,)).item()](x)

    @staticmethod
    def create_multiple_views(x: torch.Tensor, num_views: int, transformation: callable):
        return torch.stack([transformation(x) for _ in range(num_views)], dim=0)


class Noise:
    @staticmethod
    # Function to add Gaussian noise
    def add_gaussian_noise(images, noise_factor):
        noisy_images = images + noise_factor * torch.randn(*images.shape)
        noisy_images = torch.clip(noisy_images, 0., 1.)
        return noisy_images
