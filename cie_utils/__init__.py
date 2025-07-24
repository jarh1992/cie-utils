# coding: utf8
import colorsys
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from pytz import timezone
from scipy.stats import gaussian_kde, iqr
from skimage import filters
from skimage.color import lab2lch, lab2rgb, rgb2lab
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

black_px = np.array([0.0, 0.0, 0.0])
BASE_DIR = Path(__file__).resolve().parent.parent


def std4elem(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Compute the element-wise standard deviation relative to the mean of
    non-zero elements in the input array.

    This function calculates the deviation of each element from the mean
    of all non-zero elements in `x`, returning an array where elements
    originally equal to zero remain zero.

    Parameters
    ----------
    x : numpy.ndarray
        A 1D NumPy array of type float64. Elements with value zero are ignored
        in the mean and standard deviation calculation.

    Returns
    -------
    numpy.ndarray
        An array of the same shape as `x` containing the standard deviation
        for each element with respect to the mean of non-zero elements.
        Zero entries remain zero.

    Examples
    --------
    >>> import numpy as np
    >>> from cie-utils import std4elem
    >>> x = np.array([1.0, 2.0, 0.0, 4.0], dtype=np.float64)
    >>> std4elem(x)
    array([1.52752523, 0.15275252, 0., 1.37477271])
    """
    x_mean = np.mean(x[x > 0])
    n = len(x[x > 0])
    c = []
    for xi in x:
        c.append(np.sqrt(np.power(xi - x_mean, 2) / (n - 1)) if xi > 0 else 0)

    return np.array(c)


def enter_str_input(text: str) -> str:
    """
    Prompts the user for a text input and ensures it is not empty.

    Parameters
    ----------
    text : str
        The message to display to the user.

    Returns
    -------
    str
        The text string entered by the user.

    Raises
    ------
    ValueError
        If the user's input is None or an empty string.
    """
    while True:
        try:
            txt = input(text)
            if txt is None or txt == "":
                raise ValueError
            return rf"{txt}"
        except ValueError:
            print("Enter a valid text")


def normalize_img(img: npt.NDArray[np.float64], rimg: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Normalizes an image based on a reference image.

    Pixel values of the input image are divided by the corresponding values
    of the reference image. NaN values resulting from division by zero are
    replaced by zero, and values are clipped to the range [0, 1].

    Parameters
    ----------
    img : npt.NDArray[np.float64]
        The image to normalize. Must be a NumPy array of type float64.
    rimg : npt.NDArray[np.float64]
        The reference image for normalization. Must be a NumPy array of type float64.

    Returns
    -------
    npt.NDArray[np.float64]
        The normalized image, with pixel values in the range [0, 1].
    """
    norm_img: npt.NDArray[np.float64] = np.zeros_like(img, dtype=np.float64)
    for c in range(3):
        norm_img[:, :, c] = np.divide(img[:, :, c], rimg[:, :, c], out=img[:, :, c], where=rimg[:, :, c] != 0)
        norm_img[:, :, c] = np.nan_to_num(norm_img[:, :, c], nan=0)

    norm_img = np.clip(norm_img, 0, 1)
    return norm_img


def min_max(img: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Normalizes an image using min-max scaling.

    This method independently normalizes each color channel of the image
    to a range of [0, 1] based on the minimum and maximum values of each channel.
    It is sensitive to outliers.

    Parameters
    ----------
    img : npt.NDArray[np.float64]
        The input image as a NumPy array of type float64, with 3 channels.

    Returns
    -------
    npt.NDArray[np.float64]
        The normalized image in the range [0, 1].
    """
    norm_img: npt.NDArray[np.float64] = np.zeros_like(img, dtype=np.float64)
    norm_img[:, :, 0] = (img[:, :, 0] - np.amin(img[:, :, 0])) / (np.amax(img[:, :, 0] - np.amin(img[:, :, 0])))
    norm_img[:, :, 1] = (img[:, :, 1] - np.amin(img[:, :, 1])) / (np.amax(img[:, :, 1] - np.amin(img[:, :, 1])))
    norm_img[:, :, 2] = (img[:, :, 2] - np.amin(img[:, :, 2])) / (np.amax(img[:, :, 2] - np.amin(img[:, :, 2])))
    return norm_img


def sd_by_px(img: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Calculates the standard deviation per RGB pixel.

    For each pixel in the image, calculates the standard deviation of its three
    color components (R, G, B).

    Parameters
    ----------
    img : npt.NDArray[np.float64]
        The input image as a NumPy array of type float64, with 3 channels.

    Returns
    -------
    npt.NDArray[np.float64]
        A 2D array containing the standard deviation of the RGB values for each pixel.
    """
    height, width, channels = img.shape
    image_array = img.reshape(-1, channels)
    std_dev_rgb: npt.NDArray[np.float64] = np.std(image_array, axis=1)
    std_dev_rgb = np.reshape(std_dev_rgb, img.shape[:-1])
    return std_dev_rgb


def sd_by_elem(img: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Calculates the standard deviation per color component (R, G, B) for each pixel.

    For each color channel (red, green, blue), this function calculates the
    standard deviation of pixels that are greater than zero. If a pixel
    is zero, its individual standard deviation is considered zero.

    Parameters
    ----------
    img : npt.NDArray[np.float64]
        The input image as a NumPy array of type float64, with 3 channels.

    Returns
    -------
    npt.NDArray[np.float64]
        An array with the same shape as the input image, where each element
        represents the standard deviation of the corresponding color component.
    """

    img_r = img[:, :, 0].flatten()
    img_g = img[:, :, 1].flatten()
    img_b = img[:, :, 2].flatten()
    std_array: npt.NDArray[np.float64] = np.zeros_like(img)
    std_array[:, :, 0] = np.reshape(std4elem(img_r), img.shape[:-1])
    std_array[:, :, 1] = np.reshape(std4elem(img_g), img.shape[:-1])
    std_array[:, :, 2] = np.reshape(std4elem(img_b), img.shape[:-1])
    return std_array


def rm_bg(img: npt.NDArray[np.float64], sd_val: float, sdimg: Any | None = None) -> npt.NDArray[np.float64]:
    """
    Removes the background from a normalized image based on a standard deviation value.

    Pixels whose value (or the value in `sdimg` if provided) is less than
    or equal to `sd_val` are considered part of the background and are set to zero.

    Parameters
    ----------
    img : npt.NDArray[np.float64]
        The normalized input image, of type float64.
    sd_val : float
        The standard deviation threshold value. Pixels with a standard
        deviation below this value are considered background.
    sdimg : Any | None, optional
        An optional standard deviation image to use as a mask.
        If None, the input image `img` is used. Defaults to None.

    Returns
    -------
    npt.NDArray[np.float64]
        The image without background, where background pixels have been set to zero
        and values have been clipped to the range [0, 255].
    """
    # mode = max(set(data), key=data.count)
    mask = (sdimg if sdimg is not None else img) > sd_val
    img_nobg: npt.NDArray[np.float64] = np.zeros_like(img)
    img_nobg[mask] = img[mask]
    img_nobg = np.clip(img_nobg, 0, 255)
    return img_nobg


def rm_bg2channel(
    chn: npt.NDArray[np.float64],
    sd_val: float,
    sdimg: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """
    Removes the background from an image channel based on a standard deviation value.

    Pixels whose value in the channel (or the value in `sdimg` if provided)
    is less than or equal to `sd_val` are considered part of the background and are set to zero.

    Parameters
    ----------
    chn : npt.NDArray[np.float64]
        The input image channel, of type float64.
    sd_val : float
        The standard deviation threshold value. Pixels with a standard
        deviation below this value are considered background.
    sdimg : npt.NDArray[np.float64] | None, optional
        An optional standard deviation image to use as a mask.
        If None, the input channel `chn` is used. Defaults to None.

    Returns
    -------
    npt.NDArray[np.float64]
        The image channel without background, where background pixels have been set to zero
        and values have been clipped to the range [0, 1].
    """
    mask = (sdimg if sdimg is not None else chn) > sd_val
    chn_nobg: npt.NDArray[np.float64] = np.zeros_like(chn)
    chn_nobg[mask] = chn[mask]
    chn_nobg = np.clip(chn_nobg, 0, 1)

    return chn_nobg


# TODO: verify utility
def img_preparation(
    img: npt.NDArray[np.float64], rimg: npt.NDArray[np.float64], sd_val: float
) -> list[npt.NDArray[np.float64]]:
    """
    Prepares an image by applying normalization, background removal, and color space transformations.

    The input image is normalized with a reference image, its background is
    removed using a standard deviation threshold, and then it is converted
    to CIELAB and CIELCh color spaces to extract their channels.

    Parameters
    ----------
    img : npt.NDArray[np.float64]
        The input image in RGB format (float64 values).
    rimg : npt.NDArray[np.float64]
        The reference image for normalization in RGB format (float64 values).
    sd_val : float
        The standard deviation threshold value for background removal.

    Returns
    -------
    list[npt.NDArray[np.float64]]
        A list containing the L, a, b channels of the LAB image, and the
        C (chroma) and H (hue) channels of the LCh image.
        If the dimensions of `img` and `rimg` do not match, returns a list
        containing only the original input image.
    """
    if img.shape == rimg.shape:
        img_norm = normalize_img(img, rimg)
        img_sd = sd_by_px(img_norm)
        img_nobg = rm_bg(img_norm, sd_val, img_sd)
        lab_img = rgb2lab(img_nobg)
        lch_img = lab2lch(lab_img)
        return [
            lab_img[:, :, 0],
            lab_img[:, :, 1],
            lab_img[:, :, 2],
            lch_img[:, :, 1],
            lch_img[:, :, 2],
        ]
    return [img]


def get_pdf(
    img: npt.NDArray[np.float64],
) -> tuple[Any, Any, Any, int]:
    """
    Calculates the Probability Density Function (PDF) for image data.

    This function filters non-zero pixels, calculates the number of bins
    using the Freedman-Diaconis rule, and estimates the PDF using
    Gaussian Kernel Density Estimation (KDE).

    Parameters
    ----------
    img : npt.NDArray[np.float64]
        The input image as a NumPy array.

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Any, int]
        - data : npt.NDArray[np.float64]
            The pixel data (non-zero values) of the image, rounded to one decimal place.
        - rng : npt.NDArray[np.float64]
            The range of values over which the density is calculated.
        - density : Any
            The density values of the estimated probability density function.
        - bins : int
            The number of bins calculated using the Freedman-Diaconis rule.
    """
    data = np.around(img[img != 0], decimals=1)
    if data.size == 0:
        # Return safe defaults for empty data
        return data, np.array([]), np.array([]), 1
    # Calculate the Interquartile Range (IQR)
    iqr_value = iqr(data, rng=(25, 75))  # IQR between the 25th and 75th percentile

    # Calculate the number of bins using the Freedman-Diaconis rule
    n = len(data)
    bin_width = 2 * iqr_value / (n ** (1 / 3))  # Bin width
    data_range = np.amax(data) - np.amin(data)  # Range of the data
    bins = int(np.ceil(data_range / bin_width))  # Number of bins
    # print("#" * 30, "BINS", bins)

    steps = (np.amax(data) + 1 - np.amin(data)) / bins
    rng = np.arange(np.amin(data), np.amax(data) + 1, step=steps)
    density = gaussian_kde(data)
    density = density(rng)
    return data, rng, density, bins


def display_images(
    imgs: list[npt.NDArray[np.float64]],
    cols: int = 1,
    fsize: tuple[int, int] = (10, 5),
    cmaps: list[str] | None = None,
    titles: list[str] | None = None,
    save_imgs: bool = False,
    output_dir: Path = Path.cwd(),
) -> None:
    """
    Displays a list of images in a grid.

    Parameters
    ----------

    imgs : list[npt.NDArray[np.float64]]
        A list of images to display.
    cols : int, optional
        The number of columns in the display grid. Defaults to 1.
    fsize : tuple[int, int], optional
        The figure size (width, height) in inches. Defaults to (10, 5).
    cmaps : list[str] | None, optional
        A list of colormaps to apply to each image. If None, no colormap is applied.
        Defaults to None.
    titles : list[str] | None, optional
        A list of titles for each image. If None, no titles are displayed.
        Defaults to None.
    save_imgs : bool, optional
        If True, saves the images to a file instead of displaying them.
        The filename is based on the first title if available,
        otherwise "test" is used. Defaults to False.
    output_dir : Path, optional
        The directory where the images will be saved if `save_imgs` is True.
        Defaults to the current working directory.
    """
    dt = datetime.now()

    if len(imgs) < cols:
        cols = len(imgs)
    if cols < 1:
        raise ValueError("Number of columns must be at least 1.")

    rows = int(np.ceil(len(imgs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=fsize)

    # Asegurar formato uniforme para axes
    axes = np.array(axes).reshape((rows, cols)) if rows * cols > 1 else np.array([axes])

    k = 0
    for i in range(rows):
        for j in range(cols):
            if k >= len(imgs):
                axes[i, j].axis("off")
                continue
            axes[i, j].imshow(imgs[k], cmap=cmaps[k] if cmaps else None)
            if titles:
                axes[i, j].set_title(titles[k])
            axes[i, j].axis("off")
            k += 1

    for ax in axes[len(imgs) :]:
        ax.axis("off")

    if save_imgs:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"display_images_{dt.timestamp():.5f}.png"
        fig.savefig(str(output_dir / filename), dpi=600)
    else:
        plt.tight_layout()
        plt.show()

    plt.close(fig)


def display_hist(
    imgs: list[npt.NDArray[np.float64]],
    cols: int,
    fsize: tuple[int, int] = (10, 2),
    titles: list[str] | None = None,
    xlabel: list[str] = None,
    ylabel: list[str] = None,
    save_imgs: bool = False,
    save_csv: bool = False,
    output_dir: Path = Path.cwd(),
) -> None:
    """
    Displays histograms and Probability Density Functions (PDFs) for a list of images.

    Parameters
    ----------
    imgs : list[npt.NDArray[np.float64]]
        A list of images (or image channels) for which to generate histograms.
    cols : int
        The number of columns in the histogram display grid.
    fsize : tuple[int, int], optional
        The figure size (width, height) in inches. Defaults to (10, 2).
    titles : list[str] | None, optional
        A list of titles for each histogram. If None, no titles are displayed.
        Defaults to None.
    xlabel : list[str], optional
        A list of labels for the x-axis of each histogram. Defaults to ["x"] for all.
    ylabel : list[str], optional
        A list of labels for the y-axis of each histogram. Defaults to ["y"] for all.
    save_imgs : bool, optional
        If True, saves the histograms to a file instead of displaying them.
        The filename is based on the first title if available,
        otherwise "test" is used. Defaults to False.
    save_csv : bool, optional
        If True, saves the density data (rng and density) of each histogram
        to a CSV file. Defaults to False.
    output_dir : Path, optional
        The directory where the images will be saved if `save_imgs` is True.
        Defaults to the current working directory.
    """
    if not imgs:
        raise ValueError("The image list is empty.")
    if cols < 1:
        raise ValueError("The number of columns must be at least 1.")

    xlabel = xlabel or ["x"] * len(imgs)
    ylabel = ylabel or ["y"] * len(imgs)
    titles = titles or [""] * len(imgs)

    rows = int(np.ceil(len(imgs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=fsize)
    axes = np.array(axes).flatten() if isinstance(axes, np.ndarray) else [axes]

    def export_csv(index: int, rng: np.ndarray, density: np.ndarray):
        filename = output_dir / f"density_{index}_{int(time() * 1000)}.csv"
        df = pd.DataFrame(zip(rng, density), columns=["x", "pdf"])
        df.to_csv(filename, index=False)
        print(filename)

    for idx, (img, ax) in enumerate(zip(imgs, axes)):
        data, rng, density, bins = get_pdf(img)
        ax.hist(data, bins=bins, density=True)
        ax.plot(rng, density)
        ax.set_xlabel(xlabel[idx])
        ax.set_ylabel(ylabel[idx])
        ax.set_title(titles[idx])
        ax.axis("on")
        if save_csv:
            export_csv(idx, rng, density)

    for ax in axes[len(imgs) :]:
        ax.axis("off")

    output_dir.mkdir(parents=True, exist_ok=True)

    if save_imgs:
        filename = f"histograms_{datetime.now().timestamp():.5f}.png"
        fig.savefig(output_dir / filename, dpi=600)
    else:
        plt.tight_layout()
        plt.show()

    plt.close(fig)


def display_plot(
    imgs: list[npt.NDArray[np.float64]],
    fsize: tuple[int, int] = (10, 2),
    cmaps: list[str] | None = None,
    titles: list[str] | None = None,
) -> None:
    """
    Displays a combination of image, histogram, and pixel frequency scatter plot
    for each image in the list.

    For each image:
    1. The original image is displayed.
    2. A histogram and its Probability Density Function (PDF) are generated.
    3. A scatter plot of pixel value frequency is created.

    Parameters
    ----------
    imgs : list[npt.NDArray[np.float64]]
        A list of images (or image channels) to visualize.
    fsize : tuple[int, int], optional
        The figure size (width, height) in inches. Defaults to (10, 2).
    cmaps : list[str] | None, optional
        A list of colormaps to apply to each image. If None, no colormap is applied.
        Defaults to None.
    titles : list[str] | None, optional
        A list of titles for each set of plots. If None, no titles are displayed.
        Defaults to None.
    """
    rows = len(imgs)
    fig, axes = plt.subplots(rows, 3, figsize=fsize)
    for i, img in enumerate(imgs):
        pixel_num = img.shape[0] * img.shape[1]
        pix_percentage = np.around(len(img[img != 0]) / pixel_num, 2) * 100
        pix_label = f"#{pixel_num}\n{pix_percentage:.2f}%"
        if rows == 1:
            axes[0].imshow(img, cmap=cmaps[i] if cmaps else None)
            axes[0].set_title(titles[i] if titles else None)

            data, rng, density, bins = get_pdf(img)
            axes[1].hist(data, bins=bins, density=True)
            axes[1].plot(rng, density, label=pix_label)
            axes[1].legend()

            data_flat = sorted(set(data.reshape(-1)))
            pixel_freq = np.array([[dts, np.count_nonzero(data == dts)] for dts in data_flat])
            axes[2].scatter(pixel_freq[:, 0], pixel_freq[:, 1], marker=".")
        else:
            axes[i][0].imshow(img, cmap=cmaps[i] if cmaps else None)
            axes[i][0].set_title(titles[i] if titles else None)

            data, rng, density, bins = get_pdf(img)
            axes[i][1].hist(data, bins=bins, density=True)
            axes[i][1].plot(rng, density, label=pix_label)
            axes[i][1].legend()

            data_flat = sorted(set(data.reshape(-1)))
            pixel_freq = np.array([[dts, np.count_nonzero(data == dts)] for dts in data_flat])
            axes[i][2].scatter(pixel_freq[:, 0], pixel_freq[:, 1], marker=".")
    plt.tight_layout()
    plt.show()


def display_2d_scatter_plot(
    imgs: list[npt.NDArray[np.float64]],
    cols: int,
    fsize: tuple[int, int] = (10, 2),
    titles: list[str] | None = None,
) -> list[npt.NDArray[np.float64]]:
    """
    Displays 2D scatter plots of pixel frequency for a list of images.

    Each scatter plot shows pixel values (x-axis) against their
    frequency of occurrence (number of pixels with that value, y-axis). Only
    non-zero pixel values are considered.

    Parameters
    ----------
    imgs : list[npt.NDArray[np.float64]]
        A list of images (or image channels) for which to generate scatter plots.
    cols : int
        The number of columns in the plot display grid.
    fsize : tuple[int, int], optional
        The figure size (width, height) in inches. Defaults to (10, 2).
    titles : list[str] | None, optional
        A list of titles for each scatter plot. If None, no titles are displayed.
        Defaults to None.

    Returns
    -------
    list[npt.NDArray[np.float64]]
        A list of NumPy arrays, where each array contains the frequency data
        of pixels ([pixel_value, frequency]) for each input image.
    """
    plot_data: list[npt.NDArray[np.float64]] = []
    if len(imgs) < cols:
        cols = len(imgs)

    if cols < 1:
        raise ValueError("Number of columns must be at least 1.")

    rows = int(np.ceil(len(imgs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=fsize)
    if 1 in (rows, cols):
        for j, img in enumerate(imgs):
            data = np.around(img[img != 0], decimals=1)
            data_flat = sorted(set(data.reshape(-1)))
            pixel_freq = np.array([[dts, np.count_nonzero(data == dts)] for dts in data_flat])
            axes[j].scatter(pixel_freq[:, 0], pixel_freq[:, 1], marker=".")
            axes[j].set_title(titles[j] if titles else None)
            plot_data.append(pixel_freq)
    else:
        k = 0
        for i in range(rows):
            for j in range(cols):
                data = np.around(imgs[k][imgs[k] != 0], decimals=1)
                data_flat = sorted(set(data.reshape(-1)))
                pixel_freq = np.array([[dts, np.count_nonzero(data == dts)] for dts in data_flat])
                axes[i][j].scatter(pixel_freq[:, 0], pixel_freq[:, 1], marker=".")
                axes[i][j].set_title(titles[k] if titles else None)
                plot_data.append(pixel_freq)
                if k + 1 == len(imgs):
                    break
                k += 1
    plt.tight_layout()
    plt.show()
    return plot_data


def display_3d_scatter_plot(
    images: list[npt.NDArray[np.float64]],
    colors: list[str] | None = None,
    title: str | None = None,
) -> None:
    """
    Creates a 3D scatter plot for classified data.

    Each dataset in `images` is represented as a series of points
    in a 3D space, useful for visualizing cluster distribution.

    Parameters
    ----------
    images : list[npt.NDArray[np.float64]]
        A list of NumPy arrays, where each array contains the coordinates
        (e.g., L, a, b from LAB) of the pixels in a cluster or group.
        Each array is expected to have 3 columns.
    colors : list[str] | None, optional
        A list of color strings for each set of points. If None,
        colors are automatically selected. Defaults to None.
    title : str | None, optional
        The title of the plot. Defaults to None.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    images = [images] if not isinstance(images, list) else images
    for i, img in enumerate(images):
        try:
            ax.scatter(img[0], img[1], img[2], c=colors[i], label=f"# px g{i}: {img.shape[1]}")
        except IndexError:
            # Consider more specific error handling if the index is out of range
            breakpoint()

    ax.set_xlabel("A")
    ax.set_ylabel("B")
    ax.set_zlabel("L")
    ax.legend()
    ax.set_title(title)
    plt.show()


def get_csv_data(
    now: str,
    data: npt.NDArray[np.float64],
    filename: str,
    folder: Path,
    col_names: list[str] | None = None,
) -> None:
    """
    Saves data to a CSV file within a specific folder structure.

    The save path is `folder/csv/now/filename.csv`.

    Parameters
    ----------
    now : str
        A timestamp string to be used as a subfolder name within 'csv'.
    data : npt.NDArray[np.float64]
        The data to save, must be a 2D NumPy array.
    filename : str
        The name of the CSV file (without the .csv extension).
    folder : Path
        The Path object of the base folder where the 'csv/now/' structure will be created.
    col_names : list[str] | None, optional
        A list of names for the CSV columns. If None, no column names will be used.
        Defaults to None.
    """
    if col_names is None:
        col_names = []
    df = pd.DataFrame(data, columns=col_names)
    filepath = folder / "csv" / now
    filepath.mkdir(parents=True, exist_ok=True)
    csv_name = str(filepath / filename) + ".csv"
    df.to_csv(csv_name, index=False)


def get_csv_data_from_images(
    imgs: list[Any],
    folder: Path,
    col_names: list[str] | None = None,
    timezone=timezone("America/Bogota"),
) -> None:
    """
    Extracts pixel frequency data from a list of images and saves them to CSV files.

    For each image in `imgs`, pixel frequencies are calculated
    (pixel values vs. number of occurrences) and saved to a CSV file.
    Files are organized by date and time within the specified `folder`.

    Parameters
    ----------
    imgs : list[Any]
        A list of tuples where each tuple contains (image_name: str, image: npt.NDArray[np.float64]).
        The image must be a NumPy array. Pixels with zero value are ignored.
    folder : Path
        The Path object of the base folder where the CSV file structure will be created.
    col_names : list[str] | None, optional
        A list of names for the CSV columns (e.g., ["pixel_value", "frequency"]).
        If None, no column names will be used. Defaults to None.
    """
    if col_names is None:
        col_names = []
    now = datetime.now(timezone).strftime("%Y-%m-%d_%H%M%S")
    for name, img in imgs:
        data = np.around(img[img != 0], decimals=1)
        data_flat = sorted(set(data.reshape(-1)))
        pixel_freq = np.array([[n, np.count_nonzero(data == n)] for n in data_flat])
        get_csv_data(now, pixel_freq, name, folder, col_names)


def extract_segmentation(img_path: str, rimg_path: str, lbl_path: str, dest: str, dsfile: str) -> None:
    """
    Extracts image segments based on label information and a dataset file.

    This function reads an image, segmentation coordinates from a label file,
    and category names from a YAML dataset file. It then creates masks,
    crops the regions of interest (ROI), and saves the segmented images and their
    references (if provided) into folders organized by category.

    Parameters
    ----------
    img_path : str
        Absolute path to the image to be segmented.
    rimg_path : str
        Absolute path to the reference image. Can be an empty string if no reference image.
    lbl_path : str
        Absolute path to the text file with segmentation label information.
    dest : str
        Absolute path to the destination folder where segmented images will be saved.
    dsfile : str
        Absolute path to the `dataset.yaml` file containing category names.
    """
    image_path = Path(rf"{img_path}")
    label_path = Path(rf"{lbl_path}")
    dest_path = Path(rf"{dest}")
    img_name = image_path.stem
    image = cv2.imread(str(image_path))
    ref_image_path = Path(rf"{rimg_path}") if rimg_path != "" else None
    ref_image = cv2.imread(str(ref_image_path)) if ref_image_path else None
    h, w = image.shape[:2]
    with open(dsfile, "r") as file:
        data = yaml.safe_load(file)
    with open(label_path) as label:
        for i, line in enumerate(label.readlines()):
            ln = line.split()
            categ = int(ln[0])
            categ_name = data["names"][categ]
            categ_folder = dest_path / categ_name
            categ_folder.mkdir(parents=True, exist_ok=True)
            seg_coords = np.array(list(map(float, ln[1:]))).reshape((-1, 1, 2))
            seg_coords[:, :, 0] = seg_coords[:, :, 0] * w
            seg_coords[:, :, 1] = seg_coords[:, :, 1] * h
            seg_coords = np.rint(seg_coords).astype(int)

            mask = np.zeros_like(image, dtype=np.uint8)
            cv2.fillPoly(mask, [seg_coords], color=(255, 255, 255))
            roi = cv2.bitwise_and(image, mask)

            x, y, w1, h1 = cv2.boundingRect(seg_coords)
            polygon_only = roi[y : y + h1, x : x + w1]
            cv2.imwrite(f"{str(categ_folder)}/{img_name}_seg_{i:03}.png", polygon_only)

            if ref_image is not None:
                if ref_image.shape != image.shape:
                    ref_image = cv2.resize(ref_image, image.shape[1::-1])
                ref_roi = cv2.bitwise_and(ref_image, mask)
                ref_polygon_only = ref_roi[y : y + h1, x : x + w1]
                cv2.imwrite(
                    f"{str(categ_folder)}/{img_name}_seg_{i:03}_ref.png",
                    ref_polygon_only,
                )


def extract_segmentation_main() -> None:
    """
    Main function for image segmentation extraction.

    Prompts the user for the necessary paths for the image, reference image,
    label file, and destination folder, then calls `extract_segmentation`.
    """
    img_path = enter_str_input("Enter the absolute path of the image: ")
    rimg_path = input("Enter the absolute path of the reference image (enter if not): ")
    label_path = enter_str_input("Enter the absolute path of the image label: ")
    dest_path = enter_str_input("Enter the absolute path of the destination folder: ")
    dataset_yaml_path = enter_str_input("Enter the absolute path of the dataset.yaml file: ")
    extract_segmentation(img_path, rimg_path, label_path, dest_path, dataset_yaml_path)


def hsv_to_rgb(h: float, s: float, v: float) -> Any:
    """
    Converts a color from HSV to RGB.

    HSV values are in the range [0, 1]. The RGB output is in the range [0, 255]
    and are integers.

    Parameters
    ----------
    h : float
        Hue component, in the range [0, 1].
    s : float
        Saturation component, in the range [0, 1].
    v : float
        Value/Brightness component, in the range [0, 1].

    Returns
    -------
    Any
        A NumPy array of integers representing the color in RGB format [R, G, B],
        with values in the range [0, 255].
    """
    color = np.array(colorsys.hsv_to_rgb(h, s, v)) * 255
    return color.astype(int)


def create_rgb_spectrum(num_steps: int) -> list[int]:
    """
    Creates an RGB color spectrum by varying the Hue in HSV space.

    The spectrum goes from a hue close to red (0.75) to red (0),
    keeping saturation and value at maximum.

    Parameters
    ----------
    num_steps : int
        The number of steps (colors) to generate in the spectrum.

    Returns
    -------
    list[int]
        A list of NumPy arrays, where each array represents an RGB color
        ([R, G, B]) with integer values in the range [0, 255].
    """
    # Values for the HSV components (Hue, Saturation, Value)
    h_values = np.linspace(0.75, 0, num_steps)
    s = 1.0  # Saturation
    v = 1.0  # Value

    # Convert HSV to RGB for each hue value
    rgb_spectrum = [hsv_to_rgb(h, s, v).astype(np.int32) for h in h_values]

    return rgb_spectrum


def false_color_scale(channel: Any, color_list: Any, ranges_list: Any) -> Any:
    """
    Applies a false color scale to a single-channel image.

    Assigns colors from a `color_list` to the pixels of `channel`
    based on the ranges specified in `ranges_list`.

    Parameters
    ----------
    channel : Any
        The single-channel image to which the color scale will be applied.
        Expected to be a 2D NumPy array.
    color_list : Any
        A list of RGB colors ([R, G, B]) to assign to the ranges.
    ranges_list : Any
        A list of tuples or lists, where each tuple represents a range `(min_val, max_val]`.
        Pixel values within this range will be mapped to the corresponding color.

    Returns
    -------
    Any
        A new image with the false color scale applied, of integer type and 3 channels.
    """
    h, w = channel.shape
    new_chroma = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            for k, r in enumerate(ranges_list):
                if r[0] is not None:
                    if r[0] < channel[i, j] <= r[1]:
                        new_chroma[i, j] = color_list[k]
    new_chroma = new_chroma.astype(np.int32)
    return new_chroma


def mix_images(img1: Any, img2: Any) -> Any:
    """
    Blends two images: an original image and a false-color image.

    Pixels in the overlay image that are not completely black (i.e.,
    do not have an RGB component sum equal to zero) replace the
    corresponding pixels in the original image. If a pixel in `img2` is
    black, the original image's pixel is retained.

    Parameters
    ----------
    img1 : Any
        The original image (e.g., RGB). A NumPy array with 3 channels is expected.
    img2 : Any
        The image to overlay. A NumPy array with 3 channels is expected.

    Returns
    -------
    Any
        The blended image, where non-black pixels from `img2` have replaced
        pixels from `img1`.
    """
    h, w, _ = img1.shape
    new_img = np.zeros(img1.shape, dtype=np.float64)
    for i in range(h):
        for j in range(w):
            if sum(img2[i, j]) != 0:
                new_img[i, j] = img2[i, j]
            else:
                new_img[i, j] = img1[i, j]
    return new_img


def sort_classifier_results(image, labels, centers, ordered, n_ref, verbose=False):
    """
    Sorts clustering results (e.g., K-means) based on
    the number of pixels or closeness to LAB reference values.

    Parameters
    ----------
    image : npt.NDArray
        The original image (flattened if used for clustering) in LAB color space.
    labels : npt.NDArray
        The cluster identifiers assigned to each pixel.
    centers : npt.NDArray
        The coordinates (centroids) of the clusters in LAB color space.
    ordered : str
        The sorting method to use.
        Must be "by_pixel" (by number of pixels in each cluster, descending)
        or "reference_values" (by closeness to predefined LAB reference values).
    n_ref : int
        Index of the reference points to use if `ordered` is "reference_values".
        Must be 0 or 1, selecting between two sets of LAB references.
    verbose : bool, optional
        If True, prints detailed information about the sorting process
        and the average colors of the clusters. Defaults to False.

    Returns
    -------
    tuple[npt.NDArray, list[npt.NDArray]]
        - centers : npt.NDArray
            The sorted cluster centroids.
        - ordered_images : list[npt.NDArray]
            A list of images, where each element is the segmented image
            corresponding to a cluster, sorted according to the specified method.

    Raises
    ------
    ValueError
        If the `ordered` sorting method is invalid.
    """
    segmented_data = centers[labels]
    segmented_image = segmented_data.reshape(image.shape)

    images = []
    pixel_num = []

    for k in range(centers.shape[0]):
        result_image = np.zeros_like(image)
        for f in range(image.shape[0]):
            if np.allclose(segmented_image[f], centers[k]):
                result_image[f] = image[f]

        pixel_num.append(np.count_nonzero(result_image))
        images.append(result_image)

    if ordered == "by_pixel":
        # Original behavior
        indices_ordenados = np.argsort(pixel_num)[::-1]
        centers = centers[indices_ordenados]
        ordered_images = sorted(images, key=lambda li: np.count_nonzero(li), reverse=True)

    elif ordered == "reference_values":
        ref_points = [
            np.array([[82.50, 13.58, -3.27], [92.24, 14.87, -5.85], [82.50, 13.58, -3.27]]),
            np.array([[30.69, 28.91, -48.47], [60.29, 18.98, -14.80], [81.24, 14.85, -2.15]]),
        ]

        mean_colors = np.array([np.mean(img[img.any(axis=1)], axis=0) for img in images])

        if verbose:
            print("\n--- Average LAB colors of each cluster ---")
            for idx, color in enumerate(mean_colors):
                print(f"Cluster {idx}: L={color[0]:.2f}, a={color[1]:.2f}, b={color[2]:.2f}")
            print("---------------------------------------------\n")

        ordered_indexes = []
        available_indexes = list(range(len(mean_colors)))

        for i, ref in enumerate(ref_points[n_ref]):
            distances = [np.linalg.norm(mean_colors[j] - ref) for j in available_indexes]
            closest_idx_in_available = np.argmin(distances)
            assigned_cluster = available_indexes[closest_idx_in_available]

            if verbose:
                print(
                    f"Reference {i + 1}: Assigned Cluster {assigned_cluster}"
                    f" (Distance: {distances[closest_idx_in_available]:.4f})"
                )

            ordered_indexes.append(assigned_cluster)
            available_indexes.pop(closest_idx_in_available)

        ordered_indexes.append(available_indexes[0])

        # TODO: Review reference
        indices_ordenados = [ordered_indexes[i] for i in [2, 3, 1, 0]] if n_ref == 1 else ordered_indexes

        # The last cluster
        if verbose:
            print(f"Last cluster assigned: Cluster {available_indexes[0]}\n")
            print(indices_ordenados)

        centers = centers[indices_ordenados]
        ordered_images = [images[i] for i in indices_ordenados]

    else:
        raise ValueError(f"Invalid order method: {ordered}. Use 'by_pixel' or 'reference_values'.")

    return centers, ordered_images


def classifier_model(clf_model: str, params: tuple, order: str = "by_pixel", n_ref: int = 1) -> tuple[Any, Any]:
    """
    Applies a classification model (K-means, GaussianMixture, or AgglomerativeClustering)
    to images and sorts the results.

    Parameters
    ----------
    clf_model : {"Kmeans", "GaussianMixture", "AgglomerativeClustering"}
        The type of classification model to use.

    params : tuple
        Specific parameters for the classification model.

        - For "Kmeans": (img, k, stop_criteria, number_of_attempts, centroid_initialization_strategy)
        - For "GaussianMixture": (img, k)
        - For "AgglomerativeClustering": (img, k, linkage)

        `img` can be a single image or a list of images.

    order : {"by_pixel", "reference_values"}, optional
        Method to sort the resulting clusters.
        - "by_pixel": sorts by the number of pixels in each cluster (descending).
        - "reference_values": sorts by the closeness of centroids to reference LAB values.
        Defaults to "by_pixel".

    n_ref : int, optional
        Reference number (0 or 1) to select the set of reference LAB values
        when `order` is "reference_values". Defaults to 1.

    Returns
    -------
    tuple of list
        A tuple with:

        - centers : list of ndarray
            The sorted cluster centroids (or means) for each image.

        - ordered_images : list of list of ndarray
            A list of lists, where each sublist contains the segmented images
            sorted by cluster for the corresponding image.

    Raises
    ------
    ValueError
        If the `clf_model` classifier type is invalid.
    """

    images = params[0]
    images = [images] if not isinstance(images, list) else images
    lbl_cen = []

    if clf_model == "Kmeans":
        k, stop_criteria, number_of_attempts, centroid_initialization_strategy = params[1:]
        for img in images:
            _, labels, centers = cv2.kmeans(
                img,
                k,
                None,
                stop_criteria,
                number_of_attempts,
                centroid_initialization_strategy,
            )
            lbl_cen.append((labels, centers))
    elif clf_model == "GaussianMixture":
        k = params[1]
        for img in images:
            gm = GaussianMixture(n_components=k, init_params="kmeans", random_state=0).fit(img)
            labels = gm.predict(img)
            centers = gm.means_
            lbl_cen.append((labels, centers))

    elif clf_model == "AgglomerativeClustering":
        k, linkage = params[1], params[-1]
        for img in images:
            agglom = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit(img)
            labels = agglom.labels_
            # Compute centers as the mean of the points in each cluster
            centers = []
            for cluster_label in range(k):
                cluster_points = img[labels == cluster_label]
                center = np.mean(cluster_points, axis=0)
                centers.append(center)
            lbl_cen.append((labels, np.array(centers)))

    else:
        print("Invalid classifier type")
        return [list()], [[list(), list()]]

    scr = [sort_classifier_results(img, lc[0], lc[1], order, n_ref) for img, lc in zip(images, lbl_cen)]
    centers, ordered_images = zip(*scr)
    return centers, ordered_images


def plot_rgb_3d(images: Any, colors: Any, title: Any) -> None:
    """
    Generates a 3D scatter plot of RGB pixels.

    Allows visualizing the distribution of colors in 3D space (R, G, B).

    Parameters
    ----------
    images : Any | list[Any]
        An image or a list of images. Images should be NumPy arrays
        representing pixel data (e.g., RGB). If it's a 3D image
        (height, width, 3), it will be flattened to (N, 3).
    colors : Any | list[Any]
        The color or a list of colors for the points in the scatter plot.
        Must match the number of images if it's a list.
        # Example: colors = [np.array([255, 0, 0]), np.array([0, 255, 0]), np.array([0, 0, 255])]
    title : Any | list[Any]
        The title or a list of titles for the plot(s).
        Must match the number of images if it's a list.
    """
    images = images if isinstance(images, list) else [images]
    colors = colors if isinstance(colors, list) else [colors]
    title = title if isinstance(title, list) else [title]
    fig, ax = plt.subplots(1, len(images), figsize=(5 * len(images), 5), subplot_kw={"projection": "3d"})
    ax = np.array([ax]) if not isinstance(ax, np.ndarray) else ax
    for i, na in enumerate(zip(images, colors, ax)):
        img, color, ax = na
        if len(img.shape) > 2:  # if it's a tensor
            img = img.reshape(-1, 3)  # flatten to a 2d array
        arr = img.T
        try:
            ax.scatter(arr[1], arr[2], arr[0], c=color, label=f"# px g0: {img.shape[0]}")  # RGB 0-1
        except ValueError:
            # Consider more specific error handling
            breakpoint()
        ax.set_xlabel("A")
        ax.set_ylabel("B")
        ax.set_zlabel("L")
        ax.legend()
        ax.set_title(title[i])
    plt.show()


def blur_img(image: npt.NDArray) -> npt.NDArray[np.float64]:
    """
    Applies a Gaussian filter to an image.

    Uses `skimage.filters.gaussian` with a `sigma` of 1 and `truncate` of 1,
    resulting in a 3x3 Gaussian kernel.

    Parameters
    ----------
    image : npt.NDArray
        The input image. Pixel values are expected to be
        in the range [0.0, 1.0] (float) or [0, 255] (uint8).
        The `skimage.filters.gaussian` function automatically handles data types.

    Returns
    -------
    npt.NDArray[np.float64]
        The image with the Gaussian filter applied. The output will be of type float64.
        If the input was RGB, the output will also be RGB.
    """
    return filters.gaussian(image, channel_axis=-1, truncate=1)


def clahe_img(image: npt.NDArray, illumination_axis: int, top_val: float) -> npt.NDArray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to the image, specifically to the illumination channel.

    The illumination channel is scaled to [0, 255] before applying CLAHE, and then
    rescaled back to its original range [0, `top_val`].

    Parameters
    ----------
    image : npt.NDArray
        The input image. Expected to be an image in a color space
        where one channel represents illumination (e.g., the L channel of LAB).
    illumination_axis : int
        The index of the channel representing illumination in the image (e.g., 0 for L in LAB).
    top_val : float
        The maximum value the illumination channel can take in its original range
        (e.g., 100 for L in LAB).

    Returns
    -------
    npt.NDArray
        The image with CLAHE applied to the illumination channel, maintaining float64 data type.
    """
    l_ax = image[:, :, illumination_axis]
    l_ax *= 255 / top_val
    l_ax = l_ax.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(10, 10))
    l_ax_eq = clahe.apply(l_ax)  # CLAHE range is [0,255] (cv2 range), from lab/skimage.lab L range [0,100]
    l_ax_eq = l_ax_eq * (top_val / 255)
    image_eq = image.copy()
    image_eq[:, :, illumination_axis] = l_ax_eq  # float64
    return image_eq


def pca_img(image: npt.NDArray, comp_num: int) -> npt.NDArray[np.float64]:
    """
    Performs Principal Component Analysis (PCA) on the pixels of an image.

    Transforms each pixel (interpreted as a point in a 3D space, e.g., RGB or LAB)
    to its 3 principal components.

    Parameters
    ----------
    image : npt.NDArray
        The input image. A NumPy array with pixels expected
        in the last dimension (e.g., (height, width, channels)).
    comp_num : int
        The number of principal components to retain.

    Returns
    -------
    npt.NDArray[np.float64]
        A NumPy array representing the transformed image in PCA space,
        where each pixel now has 3 values corresponding to its principal components.
        The shape will be (N, 3) where N is the total number of pixels.
    """
    from sklearn.decomposition import (  # This import is crucial for the function and test
        PCA,
    )

    pca = PCA(n_components=comp_num)
    image_pca = pca.fit_transform(image)
    return image_pca


def transform_img(
    image: npt.NDArray[np.float64],
    bg_pixel: npt.NDArray[np.float64] | None = None,
    blur: bool = True,
    lab: bool = True,
    clahe: bool = True,
) -> list[npt.NDArray[np.float64]]:
    """
    Applies a series of transformations to an image: Gaussian blur,
    conversion to L*a*b* color space, and adaptive histogram equalization (CLAHE).

    Pixels that are black (considered background) can be replaced by a
    specified `bg_pixel` before transformations.

    Parameters
    ----------
    image : npt.NDArray[np.float64]
        The normalized input image, in the range [0.0, 1.0]. RGB format is expected.
    bg_pixel : npt.NDArray[np.float64] | None, optional
        An RGB pixel (float64) to insert as background where the original image
        was black. If None, black pixels are retained. Defaults to None.
    blur : bool, optional
        If True, applies a Gaussian blur to the image. Defaults to True.
    lab : bool, optional
        If True, converts the image to CIELAB color space. Defaults to True.
    clahe : bool, optional
        If True, applies CLAHE to the illumination channel of the image (only if `lab` is True).
        Defaults to True.

    Returns
    -------
    list[npt.NDArray[np.float64]]
        A list containing:

        - blur_image : npt.NDArray[np.float64]
            The blurred image (or the original image if `blur` is False).
        - rgb_image_eq : npt.NDArray[np.float64]
            The resulting image after all transformations, converted back to RGB.
        - lab_image_eq : npt.NDArray[np.float64]
            The resulting image in CIELAB color space, flattened to (N, 3)
            where N is the total number of pixels.

    """

    # background mask (information different from that of the globule)
    bg_mask = np.all(image == black_px, axis=-1)
    if bg_pixel is not None:
        image[bg_mask] = bg_pixel

    blur_image = blur_img(image) if blur else image
    lab_image = rgb2lab(blur_image) if lab else blur_image
    lab_image_eq = clahe_img(lab_image, 0, 100) if clahe else lab_image
    rgb_image_eq = lab2rgb(lab_image_eq) if lab else lab_image_eq

    if bg_pixel is not None and lab:
        lab_bg_pixel = rgb2lab(bg_pixel)
        lab_image_eq[bg_mask] = lab_bg_pixel

    lab_image_eq = lab_image_eq.reshape(-1, 3)
    lab_image_eq = np.float32(lab_image_eq)

    return [blur_image, rgb_image_eq, lab_image_eq]
