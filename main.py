import os
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class BColors:
    HEADER = '\033[95m'
    OkBLUE = '\033[94m'
    OkCYAN = '\033[96m'
    OkGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ImageUtils:
    @staticmethod
    def load_image(filepath: str) -> np.ndarray:
        """Load an image from file as a numpy array."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        return np.array(Image.open(filepath))

    @staticmethod
    def save_image(image_array: np.ndarray, filepath: str) -> None:
        """Save a numpy array as an image file."""
        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        Image.fromarray(image_array).save(filepath)

    @staticmethod
    def normalize_image(image_array: np.ndarray) -> np.ndarray:
        """Normalize image to range [0, 255]."""
        min_val = np.min(image_array)
        max_val = np.max(image_array)
        if max_val > min_val:
            normalized = (image_array - min_val) / (max_val - min_val) * 255
            return normalized
        else:
            return np.zeros_like(image_array)

    @staticmethod
    def pad_image(
            image: np.ndarray,
            padded_image_size: List[int],
            is_centered: bool = False
    ) -> np.ndarray:
        original_image_shape = image.shape
        original_image_height = original_image_shape[0]
        original_image_width = original_image_shape[1]
        # Initialize with the same dtype as the original image
        new_image_array = np.zeros(padded_image_size, dtype=image.dtype)
        if not is_centered:
            new_image_array[:original_image_height, :original_image_width] = image
        else:
            height_offset = (padded_image_size[0] - original_image_height) // 2
            width_offset = (padded_image_size[1] - original_image_width) // 2
            new_image_array[
            height_offset:height_offset + original_image_height,
            width_offset:width_offset + original_image_width
            ] = image
        return new_image_array

    @staticmethod
    def center_frequency_spectrum(image: np.ndarray) -> np.ndarray:
        """Multiply an image by (-1)^(x+y) to center the frequency spectrum."""
        height, width = image.shape[:2]
        y_coords, x_coords = np.mgrid[:height, :width]
        mask = np.where((x_coords + y_coords) % 2 == 0, 1, -1)
        centered_image = image * mask
        return centered_image

    @staticmethod
    def generate_frequency_spectrum(image: np.ndarray) -> np.ndarray:
        """Compute the normalized magnitude frequency spectrum for display."""
        fourier_transform = np.fft.fft2(image)
        magnitude_spectrum = np.abs(fourier_transform)
        # Use log scaling to compress dynamic range
        log_spectrum = np.log1p(magnitude_spectrum)
        return ImageUtils.normalize_image(log_spectrum)

    @staticmethod
    def crop_image(image: np.ndarray, locations: List[Tuple[int]]) -> np.ndarray:
        start_locations, end_locations = locations
        return image[start_locations[0]:end_locations[0], start_locations[1]:end_locations[1]]

class ImageFilter:
    """Class for different image filtering operations."""

    @staticmethod
    def gaussian_lowpass_filter(shape: Tuple[int, int], sigma: float) -> np.ndarray:
        """
        Generates a centered Gaussian Lowpass Filter for an image of size 'shape'
        with the given sigma.
        """
        P, Q = shape
        U, V = np.indices((P, Q))
        center_u, center_v = P // 2, Q // 2
        D_sq = (U - center_u) ** 2 + (V - center_v) ** 2
        H = np.exp(-D_sq / (2 * (sigma ** 2)))
        return H


class Visualizer:
    """Class for visualizing image processing results."""

    @staticmethod
    def display_images(
            images: List[np.ndarray],
            titles: List[str],
            cols: int = 2,
            cmap: str = "gray",
            file_name: str = None,
            fig_size: Tuple[int, int] = (12, 8)
    ) -> None:
        """
        Display multiple images in a single figure with given titles.
        :param file_name: filename(path) to save the plot into.
        :param images: List of images as numpy arrays.
        :param titles: List of titles for each subplot.
        :param cols: Number of columns in the subplot grid.
        :param cmap: Colormap to use.
        :param fig_size: Size of the figure.
        """
        n_images = len(images)
        rows = (n_images + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=fig_size)
        # If there is only one subplot, wrap into list.
        if rows * cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title)
            ax.axis("off")
        # Hide any empty subplots.
        for ax in axes[n_images:]:
            ax.axis("off")
        plt.tight_layout()
        if file_name is not None:
            plt.savefig(file_name)
        plt.show()


def main():
    """Process an image by padding it, centering its frequency spectrum,
    filtering in frequency domain, and displaying all steps in a single plot."""
    # -------------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------------
    print("=" * 50)
    print(f"{BColors.WARNING}{BColors.BOLD}INITIALIZATION{BColors.ENDC}{BColors.ENDC}")
    base_dir = os.path.join("Images")
    os.makedirs(base_dir, exist_ok=True)

    # -------------------------------------------------------------------------------
    # LOAD IMAGE
    # -------------------------------------------------------------------------------
    print("=" * 50)
    print(f"{BColors.WARNING}{BColors.BOLD}Loading Image ....{BColors.ENDC}{BColors.ENDC}")
    image_path = os.path.join(base_dir, "(a)Fig0431(d)(blown_ic_crop).tif")
    image_array = ImageUtils.load_image(filepath=image_path)
    print("\nImage Array:")
    print(image_array)
    print(f"{BColors.OkGREEN}{BColors.BOLD}\nImage Loaded.{BColors.ENDC}{BColors.ENDC}")

    # -------------------------------------------------------------------------------
    # PAD IMAGE
    # -------------------------------------------------------------------------------
    print("=" * 50)
    print(f"{BColors.WARNING}{BColors.BOLD}Padding Image ....{BColors.ENDC}{BColors.ENDC}")
    target_size = [2048, 2048]  # 1024 * 2
    padded_image = ImageUtils.pad_image(
        image=image_array,
        padded_image_size=target_size,
    )
    print("\nPadded Image Array:")
    print(padded_image.astype(np.uint8))
    padded_path = os.path.join(base_dir, "(b)padded_image.tiff")
    ImageUtils.save_image(image_array=padded_image, filepath=padded_path)
    print(f"{BColors.OkGREEN}{BColors.BOLD}\nPadded Image Saved.{BColors.ENDC}{BColors.ENDC}")

    # -------------------------------------------------------------------------------
    # CENTER FREQUENCY SPECTRUM
    # -------------------------------------------------------------------------------
    print("=" * 50)
    print(f"{BColors.WARNING}{BColors.BOLD}Centering Image Frequency Spectrum ....{BColors.ENDC}{BColors.ENDC}")
    centered_image_freq = ImageUtils.center_frequency_spectrum(image=padded_image)
    print("\nCentered Freq. Image Array:")
    print(centered_image_freq)
    centered_path = os.path.join(base_dir, "(c)image_with_centered_freq.tiff")
    ImageUtils.save_image(image_array=centered_image_freq, filepath=centered_path)
    print(f"{BColors.OkGREEN}{BColors.BOLD}\nCentered Freq Image Saved.{BColors.ENDC}{BColors.ENDC}")

    # -------------------------------------------------------------------------------
    # FREQUENCY SPECTRUM (Display Purposes)
    # -------------------------------------------------------------------------------
    print("=" * 50)
    print(f"{BColors.WARNING}{BColors.BOLD}Calculating Image Frequency Spectrum ....{BColors.ENDC}{BColors.ENDC}")
    transformed_image = ImageUtils.generate_frequency_spectrum(image=centered_image_freq)
    print("\nCalculated Freq. Spectrum Array:")
    print(transformed_image)
    transformed_path = os.path.join(base_dir, "(d)image_frequency_spectrum.tiff")
    ImageUtils.save_image(image_array=transformed_image, filepath=transformed_path)
    print(f"{BColors.OkGREEN}{BColors.BOLD}\nFreq Spectrum Calculated and Saved.{BColors.ENDC}{BColors.ENDC}")

    # -------------------------------------------------------------------------------
    # CREATE GAUSSIAN FILTER
    # -------------------------------------------------------------------------------
    print("=" * 50)
    print(f"{BColors.WARNING}{BColors.BOLD}Creating Gaussian Filter ....{BColors.ENDC}{BColors.ENDC}")
    # For filtering, we work with the Fourier transform of the centered image.
    F = np.fft.fft2(centered_image_freq)
    gaussian_filter = ImageFilter.gaussian_lowpass_filter(shape=F.shape, sigma=20)
    gaussian_filter_path = os.path.join(base_dir, "(e)gaussian_filter.tiff")
    ImageUtils.save_image(image_array=(gaussian_filter * 255), filepath=gaussian_filter_path)
    print(f"{BColors.OkGREEN}{BColors.BOLD}\nGaussian Filter Created.{BColors.ENDC}{BColors.ENDC}")

    # -------------------------------------------------------------------------------
    # FREQUENCY DOMAIN FILTERING
    # -------------------------------------------------------------------------------
    print("=" * 50)
    print(f"{BColors.WARNING}{BColors.BOLD}Filtering in Frequency Domain ....{BColors.ENDC}{BColors.ENDC}")
    F_filtered = np.fft.fft2(centered_image_freq) * gaussian_filter
    filtered_complex = np.fft.ifft2(F_filtered)
    filtered_image = np.real(filtered_complex)
    # Undo the centering multiplication (since center_frequency_spectrum is its own inverse)
    final_image_padded = ImageUtils.center_frequency_spectrum(image=filtered_image)

    final_image_padded_path = os.path.join(base_dir, "(g)final_image_padded.tiff")
    ImageUtils.save_image(image_array=final_image_padded, filepath=final_image_padded_path)
    # -------------------------------------------------------------------------------
    # CALCULATE PRODUCT OF TRANSFORMED IMAGE AND GAUSSIAN FILTER (VISUALIZATION PURPOSE)
    # -------------------------------------------------------------------------------
    filtered_image_spectrum_vis = transformed_image * gaussian_filter
    filtered_image_spectrum_vis_path = os.path.join(base_dir, "(f)filtered_image_spectrum.tiff")
    ImageUtils.save_image(image_array=filtered_image_spectrum_vis, filepath=filtered_image_spectrum_vis_path)

    print(f"{BColors.OkGREEN}{BColors.BOLD}\nFiltering in Frequency Domain Completed.{BColors.ENDC}{BColors.ENDC}")

    # -------------------------------------------------------------------------------
    # Cropping The padded final image
    # -------------------------------------------------------------------------------
    print("=" * 50)
    print(f"{BColors.WARNING}{BColors.BOLD}Cropping the Final Image ....{BColors.ENDC}{BColors.ENDC}")
    final_image = ImageUtils.crop_image(
        image=final_image_padded,
        locations=[(0, 0), image_array.shape]
    )
    final_image_path = os.path.join(base_dir, "(h)final_image.tiff")
    ImageUtils.save_image(image_array=final_image, filepath=final_image_path)

    print(f"{BColors.OkGREEN}{BColors.BOLD}\nCropping the Final Image Completed.{BColors.ENDC}{BColors.ENDC}")

    # -------------------------------------------------------------------------------
    # DISPLAY ALL RESULTS IN A SINGLE PLOT
    # -------------------------------------------------------------------------------
    print("=" * 50)
    print(f"{BColors.WARNING}{BColors.BOLD}Creating the Visualization ....{BColors.ENDC}{BColors.ENDC}")
    images = [
        image_array,  # Original image
        padded_image,  # Padded image
        np.clip(centered_image_freq, 0, 255).astype(np.uint8),  # Centered frequency image
        transformed_image,  # Normalized frequency spectrum
        gaussian_filter,  # Gaussian filter
        filtered_image_spectrum_vis,
        final_image_padded,  # Final filtered image (padded)
        final_image   # Final filtered image
    ]
    titles = [
        "Original Image",
        "Padded Image",
        "Centered Frequency Spectrum",
        "Frequency Spectrum (Log Scale)",
        "Gaussian Lowpass Filter",
        "Filtered Frequency Spectrum\n(Transformed * Filter)",
        "Final Filtered Image(Padded)",
        "Final Filtered Image"
    ]
    Visualizer.display_images(images, titles, cols=3, cmap="gray", fig_size=(10, 10),
                              file_name=os.path.join("Images","image_visualization.tiff"))
    print(f"{BColors.OkGREEN}{BColors.BOLD}\nCreating the Visualization Completed.{BColors.ENDC}{BColors.ENDC}")


if __name__ == '__main__':
    main()
