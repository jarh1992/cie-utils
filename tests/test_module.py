import io
import pytest
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from cie_utils import *

matplotlib.use("Agg")
# --- Test Functions ---
# Mocking external dependencies
cv2.kmeans = MagicMock(
    return_value=(
        0,
        np.array([0, 0, 1, 1]),
        np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
    )
)
cv2.boundingRect = MagicMock(return_value=(0, 0, 10, 10))
cv2.imwrite = MagicMock()
cv2.fillPoly = MagicMock()
mock_clahe_instance = MagicMock()
mock_clahe_instance.apply.return_value = np.array([[100]], dtype=np.uint8)  # Default CLAHE output
cv2.createCLAHE = MagicMock(return_value=mock_clahe_instance)

# Mock matplotlib.pyplot.show and plt.savefig
plt.show = MagicMock()
plt.savefig = MagicMock()
plt.close = MagicMock()

# Mock Path.mkdir
Path.mkdir = MagicMock()


# Mock for `input` for enter_str_input
def mock_input(prompt):
    if "absolute path" in prompt:
        return "/dummy/path"
    return "test_input"


def test_enter_str_input():
    # This function requires mocking `input`
    with patch("builtins.input", side_effect=["valid input"]):
        assert enter_str_input("Enter text: ") == "valid input"

    with patch("builtins.input", side_effect=["", "another valid input"]):
        # Redirect stdout to capture print statements
        with patch("sys.stdout", new=io.StringIO()) as fake_stdout:
            assert enter_str_input("Enter text: ") == "another valid input"
            assert "Enter a valid text" in fake_stdout.getvalue()


def test_normalize_img():
    img = np.array([[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]], dtype=np.float64)
    rimg = np.array([[[10.0, 10.0, 10.0], [80.0, 100.0, 120.0]]], dtype=np.float64)
    expected_norm_img = np.array([[[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]]], dtype=np.float64)
    # Clip to [0, 1]
    expected_norm_img = np.clip(expected_norm_img, 0, 1)

    result = normalize_img(img, rimg)
    np.testing.assert_array_almost_equal(result, expected_norm_img)

    # Test with division by zero
    img_zero = np.array([[[10.0, 20.0, 30.0]]], dtype=np.float64)
    rimg_zero = np.array([[[0.0, 10.0, 0.0]]], dtype=np.float64)
    expected_zero_result = np.array(
        [[[0.0, 2.0, 0.0]]], dtype=np.float64
    )  # 10/1e-6 = huge, clipped to 1. 30/1e-6 = huge, clipped to 1.
    expected_zero_result[0, 0, 0] = 1.0  # 10/1e-6 clipped to 1
    expected_zero_result[0, 0, 2] = 1.0  # 30/1e-6 clipped to 1

    # Corrected expected_zero_result for `img / rimg_safe` and then clip
    # 10 / 1e-6 = 1e7 -> 1.0 (clipped)
    # 20 / 10.0 = 2.0 -> 1.0 (clipped)
    # 30 / 1e-6 = 3e7 -> 1.0 (clipped)
    expected_zero_result = np.array([[[1.0, 1.0, 1.0]]], dtype=np.float64)

    result_zero = normalize_img(img_zero, rimg_zero)
    np.testing.assert_array_almost_equal(result_zero, expected_zero_result)


def test_min_max():
    img = np.array([[[10.0, 20.0, 30.0], [5.0, 25.0, 15.0], [15.0, 10.0, 20.0]]], dtype=np.float64)
    # Channel 0: [10, 5, 15]
    #   -> min=5, max=15. Norm: [ (10-5)/(15-5), (5-5)/(15-5), (15-5)/(15-5) ] = [0.5, 0.0, 1.0]
    # Channel 1: [20, 25, 10]
    #   -> min=10, max=25. Norm: [ (20-10)/(25-10), (25-10)/(25-10), (10-10)/(25-10) ] = [10/15, 1.0, 0.0]
    # Channel 2: [30, 15, 20]
    #   -> min=15, max=30. Norm: [ (30-15)/(30-15), (15-15)/(30-15), (20-15)/(30-15) ] = [1.0, 0.0, 5/15]
    expected_result = np.array([[[0.5, 10 / 15, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 5 / 15]]], dtype=np.float64)
    result = min_max(img)
    np.testing.assert_array_almost_equal(result, expected_result)


def test_std4elem():
    x = np.array([10.0, 20.0, 0.0])
    expected = np.array([5.0, 5.0, 0.0])  # mean = 15, std = sqrt((5^2)/1) = 5 for non-zero
    np.testing.assert_array_almost_equal(std4elem(x), expected)


def test_sd_by_px():
    img = np.array([[[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]]], dtype=np.float64)
    # Pixel 1: std([1, 2, 3]) = 0.81649658
    # Pixel 2: std([10, 10, 10]) = 0.0
    expected_sd = np.array([[0.81649658, 0.0]], dtype=np.float64)
    result = sd_by_px(img)
    np.testing.assert_array_almost_equal(result, expected_sd)


def test_sd_by_elem():
    img = np.array([[[10.0, 20.0, 30.0], [20.0, 30.0, 40.0], [0.0, 0.0, 0.0]]], dtype=np.float64)
    result = sd_by_elem(img)
    assert result.shape == img.shape
    assert result.dtype == np.float64


def test_rm_bg():
    img = np.array([[[0.1, 0.2, 0.3], [0.5, 0.6, 0.7], [0.0, 0.0, 0.0]]], dtype=np.float64)
    sd_val = 0.4
    # img values less than or equal to 0.4 become 0
    # Pixel 1 ([0.1, 0.2, 0.3]) -> all components <= 0.4, so becomes [0,0,0]
    # Pixel 2 ([0.5, 0.6, 0.7]) -> all components > 0.4, so remains [0.5, 0.6, 0.7]
    # Pixel 3 ([0.0, 0.0, 0.0]) -> all components <= 0.4, so remains [0,0,0]

    # Mock sd_by_px to return predictable values for rm_bg
    with patch("cie_utils.sd_by_px", return_value=np.array([[0.2, 0.6, 0.0]])):
        expected_result_no_sdimg = np.array(
            [[[0.0, 0.0, 0.0], [0.5, 0.6, 0.7], [0.0, 0.0, 0.0]]], dtype=np.float64
        )
        np.testing.assert_array_almost_equal(rm_bg(img, sd_val), expected_result_no_sdimg)

        # Test with sdimg
        sdimg_custom = np.array([[0.1, 0.8, 0.0]], dtype=np.float64)  # This will be the `sdimg` used by `rm_bg`
        # mask is (sdimg > sd_val)
        # For first pixel (sdimg val 0.1) -> 0.1 > 0.4 is False. So img becomes [0,0,0]
        # For second pixel (sdimg val 0.8) -> 0.8 > 0.4 is True. So img remains [0.5, 0.6, 0.7]
        # For third pixel (sdimg val 0.0) -> 0.0 > 0.4 is False. So img becomes [0,0,0]
        expected_result_with_sdimg = np.array(
            [[[0.0, 0.0, 0.0], [0.5, 0.6, 0.7], [0.0, 0.0, 0.0]]], dtype=np.float64
        )
        np.testing.assert_array_almost_equal(rm_bg(img, sd_val, sdimg=sdimg_custom), expected_result_with_sdimg)


def test_rm_bg2channel():
    chn = np.array([[0.1, 0.5, 0.0]], dtype=np.float64)
    sd_val = 0.3

    # When sdimg is None, it uses np.std(chn_) which is std of the whole channel.
    # std([0.1, 0.5, 0.0]) is approx 0.205
    # So `mask = chn_ > 0.205`. 0.1 is F, 0.5 is T, 0.0 is F.
    # The output from `rm_bg2channel(chn, sd_val)` when `sdimg` is None should be `chn_` where values are compared
    #   against `chn_` itself (`mask = chn_ > sd_val`).
    # No, `mask = chn_ > sd_val` is correct if `sdimg` is not provided.
    # If sdimg is None, it's `mask = chn_ > sd_val`.
    # 0.1 <= 0.3 -> 0.0
    # 0.5 > 0.3 -> 0.5
    # 0.0 <= 0.3 -> 0.0
    expected_result_no_sdimg = np.array([[0.0, 0.5, 0.0]], dtype=np.float64)
    np.testing.assert_array_almost_equal(rm_bg2channel(chn, sd_val), expected_result_no_sdimg)

    # Test with sdimg
    sdimg = np.array(
        [[0.1, 0.4, 0.0]], dtype=np.float64
    )  # This sdimg is actually not used if sdimg is None in func.
    # Let's ensure the test correctly sets `sdimg` parameter

    # Test `rm_bg2channel` with `sdimg` parameter provided.
    # `mask = sdimg > sd_val` -> `mask = [0.1>0.3, 0.4>0.3, 0.0>0.3]` -> `mask = [False, True, False]`
    # `chn_` pixels where mask is False become 0.
    # 0.1 (False) -> 0.0
    # 0.5 (True) -> 0.5
    # 0.0 (False) -> 0.0
    expected_result_with_sdimg = np.array([[0.0, 0.5, 0.0]], dtype=np.float64)
    np.testing.assert_array_almost_equal(rm_bg2channel(chn, sd_val, sdimg=sdimg), expected_result_with_sdimg)


def test_get_pdf():
    img = np.array([[1.0, 2.0, 2.0, 3.0, 0.0, 0.0]], dtype=np.float64)  # Non-zero: [1.0, 2.0, 2.0, 3.0]
    data, rng, density, bins_out = get_pdf(img)

    np.testing.assert_array_almost_equal(data, np.array([1.0, 2.0, 2.0, 3.0]))
    assert bins_out > 0  # Number of bins depends on data, Freedman-Diaconis

    # Verify rng and density shapes match
    assert rng.shape == density.shape
    assert len(rng) == bins_out  # rng has one value per bin center

    # Test with empty image (all zeros)
    empty_img = np.zeros((5, 5), dtype=np.float64)
    data_empty, rng_empty, density_empty, bins_empty = get_pdf(empty_img)
    assert data_empty.size == 0
    assert rng_empty.size == 0
    assert density_empty.size == 0
    assert bins_empty == 1  # Default bins when data is empty


def test_display_images():
    imgs = [np.random.rand(10, 10, 3), np.random.rand(10, 10, 3)]
    titles = ["Image 1", "Image 2"]
    cmaps = [None, "gray"]

    # Test basic display
    with patch("matplotlib.pyplot.show") as mock_show:
        display_images(imgs, cols=2, titles=titles, cmaps=cmaps)
        mock_show.assert_called_once()
        mock_show.reset_mock()  # Reset mock for next test

    # Test save_imgs
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with patch.object(Figure, "savefig") as mock_savefig:
            display_images(
                imgs,
                cols=1,
                titles=["test1", "test2"],
                save_imgs=True,
                output_dir=tmp_path,
            )
            mock_savefig.assert_called_once()
            saved_path = mock_savefig.call_args[0][0]
            assert tmp_path.name in saved_path
            assert "display_images" in saved_path  # Check filename format


def test_display_hist():  # TODO
    imgs = [
        np.random.rand(10, 10) * 100,
        np.random.rand(10, 10) * 50,
    ]  # single channel for histograms
    titles = ["Hist 1", "Hist 2"]
    xlabel = ["Intensity", "Value"]
    ylabel = ["Frequency", "Density"]

    # Test basic display
    with patch("matplotlib.pyplot.show") as mock_show:
        display_hist(
            imgs=imgs,
            cols=2,
            titles=titles,
            xlabel=xlabel,
            ylabel=ylabel,
            save_imgs=False,
            save_csv=False,
        )
        mock_show.assert_called_once()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with (
            patch.object(Figure, "savefig") as mock_savefig,
            patch("pandas.DataFrame.to_csv") as mock_to_csv,
        ):

            display_hist(
                imgs=imgs,
                cols=2,
                titles=titles,
                xlabel=xlabel,
                ylabel=ylabel,
                save_imgs=True,
                save_csv=True,
                output_dir=tmp_path,
            )

            # Test savefig was called with valid route
            mock_savefig.assert_called_once()
            saved_path = Path(mock_savefig.call_args[0][0])
            assert saved_path.parent == tmp_path
            assert "histograms_" in saved_path.name

            # Test the CSVs were exported
            assert mock_to_csv.call_count == len(imgs)


def test_display_plot():
    imgs = [np.random.rand(10, 10) * 100, np.random.rand(10, 10) * 50]
    titles = ["Plot 1", "Plot 2"]
    cmaps = ["viridis", "plasma"]

    display_plot(imgs, titles=titles, cmaps=cmaps)
    plt.show.assert_called_once()
    plt.show.reset_mock()


def test_display_2d_scatter_plot():
    imgs = [np.random.rand(10, 10) * 100, np.random.rand(10, 10) * 50]
    titles = ["Scatter 1", "Scatter 2"]

    result = display_2d_scatter_plot(imgs, cols=2, titles=titles)
    plt.show.assert_called_once()
    plt.show.reset_mock()

    assert isinstance(result, list)
    assert len(result) == len(imgs)

    for data in result:
        assert isinstance(data, np.ndarray)
        assert data.ndim == 2 or data.size == 0
        if data.size > 0:
            assert data.shape[1] == 2  # [pixel_value, frequency]


def test_display_3d_scatter_plot():
    # Dummy data for LAB color space (L, a, b)
    img_data_1 = np.random.rand(5, 3) * 100
    img_data_2 = np.random.rand(5, 3) * 100
    images = [img_data_1, img_data_2]
    colors = ["r", "b"]
    title = "3D Scatter Plot Test"

    display_3d_scatter_plot(images, colors, title)
    plt.show.assert_called_once()
    plt.show.reset_mock()

    # Test with single image input
    display_3d_scatter_plot(img_data_1, "g", "Single 3D Plot")
    plt.show.assert_called_once()
    plt.show.reset_mock()


def test_get_csv_data():
    now_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    data = np.array([[1, 2], [3, 4]])
    filename = "test_data"
    folder = Path("/tmp/test_csv_output")  # Use /tmp for mock outputs
    col_names = ["col1", "col2"]

    with (
        patch("pandas.DataFrame.to_csv") as mock_to_csv,
        patch("pathlib.Path.mkdir") as mock_mkdir,
    ):
        get_csv_data(now_str, data, filename, folder, col_names)
        mock_mkdir.assert_called_once()
        mock_to_csv.assert_called_once()
        # Verify call arguments
        call_args, call_kwargs = mock_to_csv.call_args
        call_args_path = Path(call_args[0])
        assert folder / Path(f"csv/{now_str}/{filename}.csv") == call_args_path  # Path object comparison
        assert call_kwargs["index"] is False


def test_get_csv_data_from_images():
    imgs_data = [
        ("image_A", np.array([[10, 20, 0], [30, 0, 40]], dtype=np.float64)),
        ("image_B", np.array([[5, 0, 15], [0, 25, 0]], dtype=np.float64)),
    ]
    folder = Path("/tmp/test_csv_output_images")
    col_names = ["value", "frequency"]

    with patch("cie_utils.get_csv_data") as mock_get_csv_data:
        get_csv_data_from_images(imgs_data, folder, col_names)
        assert mock_get_csv_data.call_count == len(imgs_data)
        # Check an example call for image_A
        call1_args = mock_get_csv_data.call_args_list[0].args
        assert call1_args[2] == "image_A"
        assert call1_args[3] == folder
        assert call1_args[4] == col_names
        # Ensure data processing for pixel_freq is correct (non-zero filtering)
        # img_array for image_A: [[10, 20, 0], [30, 0, 40]]
        # non-zero unique values: [10, 20, 30, 40]
        # counts: [1, 1, 1, 1]
        np.testing.assert_array_equal(call1_args[1], np.array([[10, 1], [20, 1], [30, 1], [40, 1]]))


def test_extract_segmentation():
    # Mock file operations and cv2 for this function
    mock_image_path = "/dummy/img.png"
    mock_rimg_path = "/dummy/ref_img.png"
    mock_lbl_path = "/dummy/label.txt"
    mock_dest_path = "/dummy/output"
    mock_dsfile = "/dummy/dataset.yaml"  # No real file access for this mock

    # Mock file reads for YAML and label file
    # Direct YAML parsing won't happen if `yaml.safe_load` is mocked.
    # The `extract_segmentation` function directly calls `yaml.safe_load` on an open file object.
    # So we need to mock `yaml.safe_load` itself.
    mock_yaml_data = {"names": {0: "category_A", 1: "category_B"}}
    mock_label_content = [
        "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2",
        "1 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6",
    ]

    with (
        patch("cv2.imread", return_value=np.zeros((100, 100, 3), dtype=np.uint8)),
        patch("cv2.imwrite") as mock_imwrite,
        patch("cv2.bitwise_and", return_value=np.zeros((100, 100, 3), dtype=np.uint8)),
        patch("cv2.boundingRect", return_value=(0, 0, 50, 50)),
        patch("builtins.open", new_callable=MagicMock) as mock_open_builtin,
        patch("yaml.safe_load", return_value=mock_yaml_data),
    ):  # Mock yaml.safe_load directly

        # Set up mock_open_builtin for the label file
        mock_open_builtin.return_value.__enter__.return_value.readlines.return_value = mock_label_content
        # The `with open(...)` for dataset.yaml will be intercepted by `yaml.safe_load` mock
        # and not actually read the file. The `with open` for label file will be handled by `readlines`.

        extract_segmentation(mock_image_path, mock_rimg_path, mock_lbl_path, mock_dest_path, mock_dsfile)

        # Assertions
        # Check if folders were created (mocked by Path.mkdir)
        assert Path.mkdir.called  # At least once for dest_base_path, then for categories
        # Specific check for category folders, if needed:
        # dest_base_path = Path(mock_dest_path)
        # (dest_base_path / 'category_A').mkdir.assert_called_once() would be more precise if we tracked args to
        # mkdir.

        # Check if cv2.imwrite was called for segmented images (image and ref image for each segment)
        assert mock_imwrite.call_count == 4  # 2 segments * (img + rimg)

        # Check fillPoly call
        assert cv2.fillPoly.called


def test_extract_segmentation_main():
    # Mock input for interactive prompts
    with (
        patch("builtins.input", side_effect=mock_input),
        patch("cie_utils.extract_segmentation") as mock_extract_segmentation,
    ):
        extract_segmentation_main()
        mock_extract_segmentation.assert_called_once()
        call_args = mock_extract_segmentation.call_args[0]
        assert call_args[0] == "/dummy/path"  # img_path
        assert call_args[1] == "/dummy/path"  # rimg_path
        assert call_args[2] == "/dummy/path"  # label_path
        assert call_args[3] == "/dummy/path"  # dest_path
        assert call_args[4] == "/dummy/path"  # dataset_yaml_path


def test_hsv_to_rgb():
    # Test known values
    # Red
    np.testing.assert_array_equal(hsv_to_rgb(0.0, 1.0, 1.0), np.array([255, 0, 0]))
    # Green
    np.testing.assert_array_equal(hsv_to_rgb(1 / 3, 1.0, 1.0), np.array([0, 255, 0]))
    # Blue
    np.testing.assert_array_equal(hsv_to_rgb(2 / 3, 1.0, 1.0), np.array([0, 0, 255]))
    # White
    np.testing.assert_array_equal(hsv_to_rgb(0.0, 0.0, 1.0), np.array([255, 255, 255]))
    # Black
    np.testing.assert_array_equal(hsv_to_rgb(0.0, 0.0, 0.0), np.array([0, 0, 0]))


def test_create_rgb_spectrum():
    num_steps = 5
    spectrum = create_rgb_spectrum(num_steps)
    assert len(spectrum) == num_steps
    for color in spectrum:
        assert isinstance(color, np.ndarray)
        assert color.shape == (3,)
        assert color.dtype == np.int32
        assert np.all((color >= 0) & (color <= 255))

    # Check general order (e.g., hue decreases from 0.75 to 0)
    # The first color should be close to purple/magenta-red, the last should be pure red.
    # We can't easily assert precise color values without knowing the exact hsv_to_rgb implementation output,
    # but we can check the range.


def test_false_color_scale():
    channel = np.array([[10, 25, 40], [55, 70, 85]], dtype=np.float64)
    color_list = [np.array([255, 0, 0]), np.array([0, 255, 0]), np.array([0, 0, 255])]
    ranges_list = [(0, 30), (30, 60), (60, 90)]  # (min, max]

    # Expected:
    # 10 (range 1: [0,30)) -> [255,0,0]
    # 25 (range 1: [0,30)) -> [255,0,0]
    # 40 (range 2: [30,60)) -> [0,255,0]
    # 55 (range 2: [30,60)) -> [0,255,0]
    # 70 (range 3: [60,90]) -> [0,0,255] (because it's the last range and inclusive)
    # 85 (range 3: [60,90]) -> [0,0,255]

    expected_new_chroma = np.array(
        [
            [[255, 0, 0], [255, 0, 0], [0, 255, 0]],
            [[0, 255, 0], [0, 0, 255], [0, 0, 255]],
        ],
        dtype=np.int32,
    )

    result = false_color_scale(channel, color_list, ranges_list)
    np.testing.assert_array_equal(result, expected_new_chroma)
    assert result.dtype == np.int32
    assert result.shape == (2, 3, 3)


def test_mix_images():
    img1 = np.array([[[10, 10, 10], [20, 20, 20]], [[30, 30, 30], [40, 40, 40]]], dtype=np.int32)
    img2 = np.array([[[0, 0, 0], [50, 50, 50]], [[0, 0, 0], [60, 60, 60]]], dtype=np.int32)

    # Expected:
    # Pixel 1 (img2 is black) -> img1 pixel [10,10,10]
    # Pixel 2 (img2 is not black) -> img2 pixel [50,50,50]
    # Pixel 3 (img2 is black) -> img1 pixel [30,30,30]
    # Pixel 4 (img2 is not black) -> img2 pixel [60,60,60]
    expected_mixed_img = np.array([[[10, 10, 10], [50, 50, 50]], [[30, 30, 30], [60, 60, 60]]], dtype=np.float64)

    result = mix_images(img1, img2)
    np.testing.assert_array_equal(result, expected_mixed_img)
    assert result.dtype == np.float64


def test_sort_classifier_results_by_pixel():
    image = np.random.rand(100, 3) * 255  # Dummy image data (flattened)
    # Create labels and centers such that cluster 1 has more pixels than cluster 0
    labels = np.array([0] * 20 + [1] * 80)  # 20 pixels in cluster 0, 80 in cluster 1
    centers = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float64)  # Cluster 0, Cluster 1 centers

    # Reconstruct image based on labels and centers for correct `mean_colors` calculation within
    # sort_classifier_results
    image_reconstructed = np.zeros_like(image)
    for i in range(len(labels)):
        image_reconstructed[i] = centers[labels[i]]

    # Expected: Cluster 1 (80px) should come before Cluster 0 (20px)
    expected_centers_order = np.array([[40, 50, 60], [10, 20, 30]], dtype=np.float64)

    sorted_centers, ordered_images = sort_classifier_results(image_reconstructed, labels, centers, "by_pixel", 0)

    np.testing.assert_array_almost_equal(sorted_centers, expected_centers_order)

    # Verify the pixel counts in ordered images (non-zero pixels count)
    # A pixel with a color value like [0,0,0] is considered 'black_px'.
    # In this test, centers are not [0,0,0], so all pixels in ordered_images should be non-black.
    assert np.count_nonzero(ordered_images[0]) == 80 * 3  # 80 pixels * 3 channels
    assert np.count_nonzero(ordered_images[1]) == 20 * 3


def test_sort_classifier_results_reference_values():
    image = np.random.rand(100, 3) * 255
    labels = np.array([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25)

    # Define centers that will map predictably to ref_points[1] when sorted
    centers_for_ref1_test = np.array(
        [
            [81.24, 14.85, -2.15],  # C0 -> matches ref[1][2]
            [10, 10, 10],  # C1 -> remaining
            [30.69, 28.91, -48.47],  # C2 -> matches ref[1][0]
            [60.29, 18.98, -14.80],  # C3 -> matches ref[1][1]
        ],
        dtype=np.float64,
    )

    # Reconstruct image for correct `mean_colors` calculation
    image_reconstructed_ref1 = np.zeros_like(image)
    for i in range(len(labels)):
        image_reconstructed_ref1[i] = centers_for_ref1_test[labels[i]]

    # Expected `ordered_indexes` based on ref_points[1] and `centers_for_ref1_test`:
    # ref[1][0] (30.69, 28.91, -48.47) is closest to C2. So `ordered_indexes` starts with [2]
    # ref[1][1] (60.29, 18.98, -14.80) is closest to C3. `ordered_indexes` becomes [2, 3]
    # ref[1][2] (81.24, 14.85, -2.15) is closest to C0. `ordered_indexes` becomes [2, 3, 0]
    # Remaining: C1. `ordered_indexes` becomes [2, 3, 0, 1]

    # For `n_ref == 1`, the reordering is `[ordered_indexes[i] for i in [2, 3, 1, 0]]`
    # ordered_indexes[2] is 0
    # ordered_indexes[3] is 1
    # ordered_indexes[1] is 3
    # ordered_indexes[0] is 2
    # So, the final order of original cluster indices should be [0, 1, 3, 2]
    expected_final_order_indices = np.array([0, 1, 3, 2])

    sorted_centers, ordered_images = sort_classifier_results(
        image_reconstructed_ref1, labels, centers_for_ref1_test, "reference_values", 1
    )

    np.testing.assert_array_almost_equal(sorted_centers, centers_for_ref1_test[expected_final_order_indices])


def test_classifier_model_kmeans():
    img = np.random.rand(100, 100, 3).astype(np.float32)  # K-means expects float32
    k = 2
    stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    number_of_attempts = 10
    centroid_initialization_strategy = cv2.KMEANS_RANDOM_CENTERS

    params = (
        [img],
        k,
        stop_criteria,
        number_of_attempts,
        centroid_initialization_strategy,
    )

    # Mock cv2.kmeans to return predictable results
    mock_labels = np.array([0] * 50 + [1] * 50)
    mock_centers = np.array([[10, 10, 10], [20, 20, 20]], dtype=np.float32)
    cv2.kmeans.return_value = (0, mock_labels, mock_centers)

    with patch(
        "cie_utils.sort_classifier_results",
        side_effect=lambda img, labels, centers, order, n_ref: (centers, [img, img]),
    ):
        centers, ordered_images = classifier_model("Kmeans", params, order="by_pixel")
        assert len(centers) == 1
        assert len(ordered_images[0]) == 2  # Because sort_classifier_results returns two dummy images
        np.testing.assert_array_almost_equal(centers[0], mock_centers)
        cv2.kmeans.assert_called_once_with(
            img,
            k,
            None,
            stop_criteria,
            number_of_attempts,
            centroid_initialization_strategy,
        )


def test_classifier_model_gaussianmixture():
    img = np.random.rand(100, 3)
    k = 2
    params = [img, k]

    with patch("sklearn.mixture.GaussianMixture"):
        with patch(
            "cie_utils.sort_classifier_results",
            side_effect=lambda img, labels, centers, order, n_ref: (
                centers,
                [img, img],
            ),
        ):
            centers, ordered_images = classifier_model("GaussianMixture", params)
            assert len(centers) == 1
            assert len(ordered_images[0]) == 2


def test_classifier_model_agglomerativeclustering():
    img = np.random.rand(100, 3)
    k = 2
    linkage = "ward"
    params = [img, k, linkage]

    # Mock AgglomerativeClustering
    with patch("sklearn.cluster.AgglomerativeClustering"):
        with patch(
            "cie_utils.sort_classifier_results",
            side_effect=lambda img, labels, centers, order, n_ref: (
                centers,
                [img, img],
            ),
        ):
            centers, ordered_images = classifier_model("AgglomerativeClustering", params)
            assert len(centers) == 1
            assert len(ordered_images[0]) == 2


def test_classifier_model_unknown_cluster_method():
    img = np.random.rand(100, 3)
    k = 2
    params = [img, k]
    with patch(
        "cie_utils.sort_classifier_results",
        side_effect=lambda img, labels, centers, order, n_ref: (centers, [img, img]),
    ):
        centers, ordered_images = classifier_model("unknown", params, order="by_pixel")
        assert len(centers) == 1
        assert len(ordered_images[0]) == 2


def test_plot_rgb_3d():
    images = [
        np.random.rand(10, 10, 3) * 255,
        np.random.rand(10, 3) * 255,
    ]  # 3D image and 2D pixel array
    colors = ["r", "b"]
    titles = ["Plot 1", "Plot 2"]

    plot_rgb_3d(images, colors, titles)
    plt.show.assert_called_once()
    plt.show.reset_mock()

    # Test with single image input
    single_image = np.random.rand(5, 5, 3) * 255
    single_color = "g"
    single_title = "Single RGB Plot"
    plot_rgb_3d(single_image, single_color, single_title)
    plt.show.assert_called_once()
    plt.show.reset_mock()


def test_blur_img():
    # Create a simple image with distinct values to see blurring effect
    image = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )

    # Mock skimage.filters.gaussian to return a predictable blurred image
    mock_blurred = np.array(
        [
            [[0.1, 0.1, 0.1], [0.5, 0.5, 0.5], [0.1, 0.1, 0.1]],
            [[0.1, 0.1, 0.1], [0.5, 0.5, 0.5], [0.1, 0.1, 0.1]],
            [[0.1, 0.1, 0.1], [0.5, 0.5, 0.5], [0.1, 0.1, 0.1]],
        ],
        dtype=np.float64,
    )

    with patch("skimage.filters.gaussian", return_value=mock_blurred) as mock_gaussian:
        result = blur_img(image)
        np.testing.assert_array_equal(result, mock_blurred)
        mock_gaussian.assert_called_once_with(image, channel_axis=-1, truncate=1)


def test_clahe_img():
    # Create a dummy LAB image (L channel, a, b)
    # L channel values in [0, 100]
    image = np.array(
        [
            [[50.0, 10.0, -20.0], [70.0, 5.0, 15.0]],
            [[20.0, -5.0, 30.0], [90.0, 20.0, -10.0]],
        ],
        dtype=np.float64,
    )
    illumination_axis = 0  # L channel
    top_val = 100.0  # Max L value

    # Calculate the exact expected input that should be passed to clahe.apply()
    # This is the image's L channel scaled to [0, 255] and converted to uint8.
    expected_input_to_clahe_apply = (image[:, :, illumination_axis] * (255 / top_val)).astype(np.uint8)
    # For the provided 'image', this evaluates to: [[127, 178], [51, 229]]

    # Define the mock *output* from clahe.apply().
    # This should be distinct from the input, representing a processed image.
    # These are arbitrary example values in the range [0, 255].
    mock_clahe_returned_value = np.array(
        [[160, 210], [80, 250]],  # Arbitrary values representing CLAHE output
        dtype=np.uint8,
    )

    # Mock cv2.createCLAHE and clahe_instance.apply
    mock_clahe_instance = MagicMock()
    mock_clahe_instance.apply.return_value = mock_clahe_returned_value
    with patch("cv2.createCLAHE", return_value=mock_clahe_instance) as mock_createCLAHE:
        result = clahe_img(image, illumination_axis, top_val)

        # Calculate the expected L channel in the final result,
        # based on the mock_clahe_returned_value, scaled back to [0, 100].
        expected_l_eq = mock_clahe_returned_value * (top_val / 255)
        # For mock_clahe_returned_value = [[160, 210], [80, 250]]:
        # This calculates to:
        # [[62.745098, 82.352941],
        #  [31.372549, 98.039216]]

        # Verify the L channel is updated correctly based on the mocked output
        np.testing.assert_array_almost_equal(result[:, :, illumination_axis], expected_l_eq, decimal=6)
        # Verify other channels remain unchanged
        np.testing.assert_array_equal(result[:, :, 1], image[:, :, 1])  # 'a' channel
        np.testing.assert_array_equal(result[:, :, 2], image[:, :, 2])  # 'b' channel
        # Verify the data type of the final result
        assert result.dtype == np.float64

        # Verify cv2.createCLAHE was called with the correct parameters
        mock_createCLAHE.assert_called_once_with(clipLimit=5, tileGridSize=(10, 10))

        # IMPORTANT FIX: Verify that the input to clahe_instance.apply was correctly scaled
        # This asserts that the actual argument passed to .apply() matches our expected calculation.
        np.testing.assert_array_equal(
            mock_clahe_instance.apply.call_args[0][0],  # The actual argument passed to .apply()
            expected_input_to_clahe_apply,  # Our calculated expected input
        )


def test_pca_img():
    # Create a dummy image (e.g., flattened LAB image)
    image = np.array([[10, 20, 30], [12, 22, 33], [5, 10, 15], [7, 12, 18]], dtype=np.float64)
    comp_num = 2

    # Mock PCA
    mock_pca_instance = MagicMock()
    mock_pca_instance.fit_transform.return_value = np.array(
        [
            [7.3441764, -0.55952928],
            [11.36306385, 0.36163501],
            [-11.36306385, -0.36163501],
            [-7.3441764, 0.55952928],
        ],
        dtype=np.float64,
    )

    with patch("sklearn.decomposition.PCA", return_value=mock_pca_instance) as mock_pca:
        result = pca_img(image, comp_num)
        np.testing.assert_array_almost_equal(result, mock_pca_instance.fit_transform.return_value, decimal=6)
        mock_pca.assert_called_once_with(n_components=comp_num)
        mock_pca_instance.fit_transform.assert_called_once_with(image)


def test_transform_img():
    image_rgb_norm = np.array(
        [[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], [[0.0, 0.0, 0.0], [0.8, 0.8, 0.8]]],
        dtype=np.float64,
    )

    bg_pixel = np.array([0.1, 0.1, 0.1], dtype=np.float64)

    # Mock internal function calls
    mock_blur_image = np.array(
        [
            [
                [0.23216153, 0.23216153, 0.23216153],
                [0.45005906, 0.45005906, 0.45005906],
            ],
            [
                [0.26931395, 0.26931395, 0.26931395],
                [0.54846546, 0.54846546, 0.54846546],
            ],
        ],
        dtype=np.float64,
    )
    mock_lab_image = np.array(
        [
            [
                [6.36472978e01, -8.66841197e-04, 1.64312801e-03],
                [1.23285397e02, -1.36179666e-03, 2.58133353e-03],
            ],
            [
                [7.43247518e01, -9.55456768e-04, 1.81110194e-03],
                [1.48398696e02, -1.57021988e-03, 2.97640708e-03],
            ],
        ],
        dtype=np.float64,
    )
    mock_lab_image_eq = np.array(
        [
            [
                [1.00000000e02, -8.66841197e-04, 1.64312801e-03],
                [1.00000000e02, -1.36179666e-03, 2.58133353e-03],
            ],
            [
                [1.00000000e02, -9.55456768e-04, 1.81110194e-03],
                [1.00000000e02, -1.57021988e-03, 2.97640708e-03],
            ],
        ],
        dtype=np.float64,
    )  # Assuming no change by clahe for simplicity in mock

    mock_rgb_image_eq = np.array(
        [
            [[1.0, 0.999997, 1.0], [1.0, 0.999998, 1.0]],
            [[1.0, 0.999997, 1.0], [1.0, 0.999998, 1.0]],
        ],
        dtype=np.float64,
    )  # Assuming no change by clahe for simplicity in mock

    # Mock skimage.color.rgb2lab and skimage.color.lab2rgb
    with (
        patch("cie_utils.blur_img", return_value=mock_blur_image),
        patch("skimage.color.rgb2lab", return_value=mock_lab_image),
        patch("skimage.color.lab2rgb", return_value=mock_blur_image),
        patch("cie_utils.clahe_img", return_value=mock_lab_image_eq),
    ):
        # Test with all transformations ON and bg_pixel
        # Make a copy because `image` is modified in-place by `bg_pixel` logic.
        image_for_test = image_rgb_norm.copy()
        result_all_on = transform_img(image_for_test, bg_pixel=bg_pixel, blur=True, lab=True, clahe=True)

        assert len(result_all_on) == 3
        # The first output is the (mocked) blurred image.
        np.testing.assert_array_equal(result_all_on[0], mock_blur_image)
        # The second output is rgb_image_eq, which is lab2rgb(mock_lab_image_eq)
        np.testing.assert_array_almost_equal(
            result_all_on[1], mock_rgb_image_eq, decimal=6
        )  # As lab2rgb is mocked to return mock_blur_image
        # The third output is lab_image_eq flattened
        expected_lab_image_eq_flat = mock_lab_image_eq.reshape(-1, 3).astype(np.float32)
        np.testing.assert_array_equal(result_all_on[2], expected_lab_image_eq_flat)

        # Test with blur=False, lab=False, clahe=False
        image_for_test_off = image_rgb_norm.copy()
        result_all_off = transform_img(image_for_test_off, bg_pixel=bg_pixel, blur=False, lab=False, clahe=False)

        # For result_all_off, the image_for_test_off would have its black pixels replaced by bg_pixel.
        # Original: [[[0,0,0], [0.5,0.5,0.5]], [[0,0,0], [0.8,0.8,0.8]]]
        # After bg_pixel: [[[0.1,0.1,0.1], [0.5,0.5,0.5]], [[0.1,0.1,0.1], [0.8,0.8,0.8]]]
        expected_original_transformed_if_bg = np.array(
            [[[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]], [[0.1, 0.1, 0.1], [0.8, 0.8, 0.8]]],
            dtype=np.float64,
        )

        np.testing.assert_array_equal(
            result_all_off[0], expected_original_transformed_if_bg
        )  # Original image if no blur
        np.testing.assert_array_equal(
            result_all_off[1], expected_original_transformed_if_bg
        )  # Original image if no lab conversion
        # Flattened original image if no lab conversion
        np.testing.assert_array_equal(
            result_all_off[2],
            expected_original_transformed_if_bg.reshape(-1, 3).astype(np.float32),
        )

def test_find_intersections_basic():
    """Test: Find intersections between two straight lines."""
    # Line 1: y = x
    x1 = np.array([0, 10])
    y1 = np.array([0, 10])

    # Line 2: y = 10 - x
    x2 = np.array([0, 10])
    y2 = np.array([10, 0])

    intersections = find_intersections(x1, y1, x2, y2)

    # They should intersect at x=5, y=5
    assert len(intersections) == 1, f"Expected 1 intersection, found {len(intersections)}"

    x_int, y_int = intersections[0]
    assert np.isclose(x_int, 5.0, atol=0.1), f"Expected x ~5.0, got {x_int}"
    assert np.isclose(y_int, 5.0, atol=0.1), f"Expected y ~5.0, got {y_int}"


def test_find_intersections_no_overlap():
    """Test: Curves that do not intersect."""
    x1 = np.array([0, 5])
    y1 = np.array([0, 5])

    x2 = np.array([10, 15])
    y2 = np.array([10, 15])

    intersections = find_intersections(x1, y1, x2, y2)

    assert len(intersections) == 0, f"Expected 0 intersections, found {len(intersections)}"


def test_find_intersections_multiple():
    """Test: Find multiple intersections."""
    # Sine function
    x1 = np.linspace(0, 4 * np.pi, 100)
    y1 = np.sin(x1)

    # Horizontal line at y=0.5
    x2 = np.array([0, 4 * np.pi])
    y2 = np.array([0.5, 0.5])

    intersections = find_intersections(x1, y1, x2, y2)

    # Sine should cross y=0.5 multiple times between 0 and 4*pi
    assert len(intersections) >= 2, f"Expected at least 2 intersections, found {len(intersections)}"


def test_filter_intersections_for_console():
    """Test: Filter nearby intersections."""
    # Create nearby intersections
    intersections = [
        (1.0, 2.0),
        (1.2, 2.1),  # Very close to the first (distance 0.2)
        (3.0, 4.0),
        (3.1, 4.1),  # Very close to the third (distance 0.1)
        (6.0, 7.0)   # Far away
    ]

    filtered = filter_intersections_for_console(intersections, distance_threshold=0.5)

    # Only 3 should remain (first, third, and fifth)
    assert len(filtered) == 3, f"Expected 3 filtered intersections, found {len(filtered)}"

    # Verify the correct ones are kept
    x_vals = [x for x, _ in filtered]
    assert 1.0 in x_vals, "First intersection should be present"
    assert 3.0 in x_vals, "Third intersection should be present"
    assert 6.0 in x_vals, "Fifth intersection should be present"


def test_filter_intersections_for_console_empty():
    """Test: Filter an empty list of intersections."""
    filtered = filter_intersections_for_console([])
    assert len(filtered) == 0, "Empty list should return empty list"

    filtered = filter_intersections_for_console([], distance_threshold=2.0)
    assert len(filtered) == 0, "Empty list with threshold should return empty list"


def test_detect_number_of_clusters():
    """Test: Automatic detection of the number of clusters."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test label file
        test_labels = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
        labels_path = os.path.join(temp_dir, "test_labels.npy")
        np.save(labels_path, test_labels)

        # Create file with the correct extension
        final_path = os.path.join(temp_dir, "image_labels.npy")
        shutil.move(labels_path, final_path)

        # Detect number of clusters
        n_clusters = detect_number_of_clusters(temp_dir)

        # Labels have values 0, 1, 2 -> valid clusters: 1, 2 -> 2 clusters
        assert n_clusters == 2, f"Expected 2 clusters, detected {n_clusters}"


def test_detect_number_of_clusters_no_files():
    """Test: Detection when there are no files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Empty directory
        n_clusters = detect_number_of_clusters(temp_dir)

        # Should return 3 by default
        assert n_clusters == 3, f"Without files should return 3, returned {n_clusters}"


def test_combine_red_blood_cell_clusters():
    """Test: Combination of clusters 3 and 4."""
    # Create test data
    data = {
        1: {"L": [1, 2, 3], "a": [10, 11, 12], "b": [20, 21, 22]},
        2: {"L": [4, 5, 6], "a": [13, 14, 15], "b": [23, 24, 25]},
        3: {"L": [7, 8], "a": [16, 17], "b": [26, 27]},
        4: {"L": [9, 10], "a": [18, 19], "b": [28, 29]}
    }

    n_clusters = 4

    # Combine clusters
    combined_data, new_n = combine_red_blood_cell_clusters(data, n_clusters)

    # Verify results
    assert new_n == 3, f"Expected 3 clusters after combining, got {new_n}"
    assert 4 not in combined_data, "Cluster 4 should have been removed"
    assert len(combined_data[3]["L"]) == 4, "Cluster 3 should have 4 elements in L"
    assert len(combined_data[3]["a"]) == 4, "Cluster 3 should have 4 elements in a"
    assert len(combined_data[3]["b"]) == 4, "Cluster 3 should have 4 elements in b"


def test_combine_red_blood_cell_clusters_no_combine():
    """Test: Do not combine when there are fewer than 4 clusters."""
    data = {
        1: {"L": [1, 2], "a": [10, 11], "b": [20, 21]},
        2: {"L": [3, 4], "a": [12, 13], "b": [22, 23]}
    }

    n_clusters = 2

    combined_data, new_n = combine_red_blood_cell_clusters(data, n_clusters)

    # Nothing should change
    assert new_n == 2, f"Expected 2 clusters, got {new_n}"
    assert 1 in combined_data, "Cluster 1 should exist"
    assert 2 in combined_data, "Cluster 2 should exist"


def test_get_automatic_names_and_colors():
    """Test: Automatic generation of names and colors."""
    # For 2 clusters
    names_2, colors_2 = get_automatic_names_and_colors(2)

    assert len(names_2) == 2, f"Expected 2 names, got {len(names_2)}"
    assert len(colors_2) == 2, f"Expected 2 colors, got {len(colors_2)}"
    assert names_2[1] == "Chromatin", f"Cluster 1 name incorrect: {names_2[1]}"
    assert names_2[2] == "Cytoplasm", f"Cluster 2 name incorrect: {names_2[2]}"

    # For 5 clusters (more than predefined)
    names_5, colors_5 = get_automatic_names_and_colors(5)

    assert len(names_5) == 5, f"Expected 5 names, got {len(names_5)}"
    assert len(colors_5) == 5, f"Expected 5 colors, got {len(colors_5)}"
    assert names_5[5] == "Other", f"Cluster 5 name incorrect: {names_5[5]}"

    # For 7 clusters (outside predefined)
    names_7, colors_7 = get_automatic_names_and_colors(7)

    assert len(names_7) == 7, f"Expected 7 names, got {len(names_7)}"
    assert names_7[7] == "Cluster 7", f"Cluster 7 name incorrect: {names_7[7]}"


def test_compute_histograms():
    """Test: Smoothed histogram computation."""
    # Create test data
    data = {
        1: {
            "L": np.random.normal(50, 10, 1000).tolist(),
            "a": np.random.normal(0, 5, 1000).tolist(),
            "b": np.random.normal(0, 5, 1000).tolist()
        },
        2: {
            "L": np.random.normal(70, 8, 800).tolist(),
            "a": np.random.normal(10, 3, 800).tolist(),
            "b": np.random.normal(5, 4, 800).tolist()
        }
    }

    n_clusters = 2

    # Compute histograms
    curves = compute_histograms(data, n_clusters, bins=20)

    # Verify structure
    assert 'L' in curves, "Should have component L"
    assert 'a' in curves, "Should have component a"
    assert 'b' in curves, "Should have component b"

    # Verify there is data for each cluster
    for comp in ['L', 'a', 'b']:
        assert 1 in curves[comp], f"Should have data for cluster 1 in {comp}"
        assert 2 in curves[comp], f"Should have data for cluster 2 in {comp}"

        # Verify each entry is a tuple (x, y)
        x1, y1 = curves[comp][1]
        assert len(x1) > 0, f"x should not be empty for cluster 1 in {comp}"
        assert len(y1) > 0, f"y should not be empty for cluster 1 in {comp}"

        assert len(x1) == len(y1), f"x and y must have the same length for cluster 1 in {comp}"


def test_compute_all_intersections():
    """Test: Intersection computation between cluster pairs."""
    # Create test smoothed curves that actually intersect
    x = np.linspace(0, 10, 50)
    smoothed_curves = {
        'L': {
            1: (x, 5 + 5 * np.sin(x)),         # Sine wave oscillating around 5
            2: (x, np.full_like(x, 5.0))        # Horizontal line at y=5
        },
        'a': {
            1: (np.array([0, 10]), np.array([0, 10])),   # y = x (increasing)
            2: (np.array([0, 10]), np.array([10, 0]))     # y = 10-x (decreasing)
        },
        'b': {
            1: (np.array([0, 10]), np.array([0, 0])),     # Horizontal line at y=0
            2: (np.array([0, 10]), np.array([5, 5]))       # Horizontal line at y=5
        }
    }

    names = {1: "Cluster 1", 2: "Cluster 2"}

    # Compute intersections (use smaller threshold for small x-range test data)
    intersections = compute_all_intersections(smoothed_curves, names, distance_threshold=1.0)

    # Verify structure
    assert 'L' in intersections, "Should have intersections for L"
    assert 'a' in intersections, "Should have intersections for a"
    assert 'b' in intersections, "Should have intersections for b"

    # For L and a there should be intersections, for b there should not
    assert len(intersections['L']) > 0, "Should have intersections in L"
    assert len(intersections['a']) > 0, "Should have intersections in a"
    assert len(intersections['b']) == 0, "Should not have intersections in b (parallel lines)"


def test_compute_all_intersections_no_data():
    """Test: Intersection computation with no data."""
    smoothed_curves = {'L': {}, 'a': {}, 'b': {}}
    names = {}

    intersections = compute_all_intersections(smoothed_curves, names)

    for comp in ['L', 'a', 'b']:
        assert len(intersections[comp]) == 0, f"Should not have intersections in {comp}"


def create_complete_test_data(temp_dir: str, num_images: int = 3):
    """
    Create complete test data for testing.

    Parameters
    ----------
    temp_dir : str
        Temporary directory path.
    num_images : int, optional
        Number of test images to create (default is 3).

    Returns
    -------
    tuple of (str, str)
        Paths to the input directory and labels directory.
    """
    input_dir = os.path.join(temp_dir, "input")
    labels_dir = os.path.join(temp_dir, "labels")

    os.makedirs(input_dir)
    os.makedirs(labels_dir)

    for i in range(num_images):
        # Create random LAB image
        image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8).astype(np.float32)

        # Create labels with 3 clusters (0, 1, 2, 3)
        labels = np.random.choice([0, 1, 2, 3], size=(10, 10))

        # Save files
        image_path = os.path.join(input_dir, f"image_{i}.npy")
        labels_path = os.path.join(labels_dir, f"image_{i}_labels.npy")

        np.save(image_path, image)
        np.save(labels_path, labels)

    return input_dir, labels_dir


def test_analyze_lab_distributions_basic():
    """Test: Basic LAB distribution analysis."""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir, labels_dir = create_complete_test_data(temp_dir, num_images=2)

        # Run analysis without plots
        results = analyze_lab_distributions(
            input_dir, labels_dir,
            use_sampling=False,
            show_plots=False,
            bins=20
        )

        # Verify results
        assert 'data' in results, "Results should contain 'data'"
        assert 'n_clusters' in results, "Results should contain 'n_clusters'"
        assert 'smoothed_curves' in results, "Results should contain 'smoothed_curves'"
        assert 'intersections_by_component' in results, "Results should contain 'intersections_by_component'"

        # Verify detected number of clusters
        # Our labels have values 0, 1, 2, 3 -> valid clusters: 1, 2, 3
        assert results['n_clusters'] == 3, f"Expected 3 clusters, got {results['n_clusters']}"

        # Verify that images were processed
        assert results['num_images_processed'] == 2, \
            f"Expected 2 processed images, got {results['num_images_processed']}"


def test_analyze_lab_distributions_with_sampling():
    """Test: Analysis with random sampling."""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir, labels_dir = create_complete_test_data(temp_dir, num_images=2)

        # Run analysis with sampling
        results = analyze_lab_distributions(
            input_dir, labels_dir,
            use_sampling=True,
            sampling_fraction=0.5,
            show_plots=False,
            bins=20
        )

        # Verify that sampling was used
        assert results['use_sampling'] == True, "Should indicate that sampling was used"
        assert results['sampling_fraction'] == 0.5, f"Incorrect sampling fraction: {results['sampling_fraction']}"


def test_analyze_lab_distributions_no_files():
    """Test: Analysis with no files (should fail)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Empty directory
        with pytest.raises(ValueError, match="No label files found"):
            analyze_lab_distributions(
                temp_dir, temp_dir,
                show_plots=False
            )


def test_analyze_lab_distributions_invalid_paths():
    """Test: Analysis with invalid paths."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create only images, no labels
        input_dir = os.path.join(temp_dir, "input")
        labels_dir = os.path.join(temp_dir, "labels")

        os.makedirs(input_dir)

        # Create an image without corresponding labels
        image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8).astype(np.float32)
        np.save(os.path.join(input_dir, "image_0.npy"), image)

        # labels_dir does not exist, so detect_number_of_clusters will warn
        # and then analyze_lab_distributions will raise ValueError (no label files)
        with pytest.warns(UserWarning, match="Labels directory does not exist"):
            with pytest.raises(ValueError, match="No label files found"):
                analyze_lab_distributions(
                    input_dir, labels_dir,
                    show_plots=False
                )

def test_median_filter_invalid_kernel():
    """Test: kernel_size must be odd and positive."""
    matrix = np.ones((5, 5))

    # Even kernel_size should raise an error
    with pytest.raises(ValueError, match="kernel_size must be odd"):
        apply_median_filter(matrix, kernel_size=4)

    # kernel_size <= 0 should raise an error
    with pytest.raises(ValueError, match="kernel_size must be greater than 0"):
        apply_median_filter(matrix, kernel_size=0)

    with pytest.raises(ValueError, match="kernel_size must be greater than 0"):
        apply_median_filter(matrix, kernel_size=-1)


def test_median_filter_ordered_matrix():
    """Test: Median filter on an ordered matrix with 3x3 kernel."""
    matrix = create_test_matrix(5, 'ordered')

    # Apply filter
    result = apply_median_filter(matrix, kernel_size=3)

    # Verify center position (2,2)
    # 3x3 neighborhood centered at (2,2): [[6,7,8],[11,12,13],[16,17,18]]
    # Median = 12
    assert np.isclose(result[2, 2], 12.0), \
        f"Position (2,2) expected 12.0, got {result[2, 2]}"

    # Verify position (1,1)
    # Neighborhood: [[0,1,2],[5,6,7],[10,11,12]], Median = 6
    assert np.isclose(result[1, 1], 6.0), \
        f"Position (1,1) expected 6.0, got {result[1, 1]}"


def test_median_filter_random_matrix():
    """Test: Median filter on a random matrix."""
    np.random.seed(42)
    matrix = create_test_matrix(7, 'random')

    # Apply filter with 3x3 kernel
    result = apply_median_filter(matrix, kernel_size=3)

    # Verify some positions manually
    positions_to_verify = [(3, 3), (0, 0), (6, 6), (2, 5)]

    for i, j in positions_to_verify:
        manual = compute_manual_median(matrix, i, j, kernel_size=3)
        filtered = result[i, j]

        assert np.isclose(manual, filtered, rtol=1e-7), \
            f"Position ({i},{j}): manual={manual}, filtered={filtered}"


def test_median_filter_kernel_5x5():
    """Test: Median filter with 5x5 kernel."""
    matrix = create_test_matrix(10, 'ordered')

    # Apply filter with 5x5 kernel
    result = apply_median_filter(matrix, kernel_size=5)

    # Verify position (5,5) - center of the matrix
    # Ordered 10x10 matrix, value at (5,5) = 55
    # 5x5 neighborhood: values 33-43-53-63-73, 34-44-54-64-74, etc.
    # All values: from 33 to 77, Median = 55
    assert np.isclose(result[5, 5], 55.0), \
        f"Position (5,5) with 5x5 kernel expected 55.0, got {result[5, 5]}"

    # Verify with manual computation
    manual = compute_manual_median(matrix, 5, 5, kernel_size=5)
    assert np.isclose(result[5, 5], manual), \
        f"Discrepancy at (5,5): filtered={result[5, 5]}, manual={manual}"


def test_median_filter_rgb():
    """Test: Median filter for RGB images."""
    # Create 5x5x3 RGB image
    np.random.seed(123)
    channel_r = create_test_matrix(5, 'random')
    channel_g = create_test_matrix(5, 'ordered')
    channel_b = create_test_matrix(5, 'stepped')

    rgb_image = np.stack([channel_r, channel_g, channel_b], axis=-1)

    # Apply filter
    result = apply_median_filter_rgb(rgb_image, kernel_size=3)

    # Verify that each channel was processed independently
    result_r = apply_median_filter(channel_r, kernel_size=3)
    result_g = apply_median_filter(channel_g, kernel_size=3)
    result_b = apply_median_filter(channel_b, kernel_size=3)

    assert np.allclose(result[:, :, 0], result_r), \
        "R channel does not match individual processing"
    assert np.allclose(result[:, :, 1], result_g), \
        "G channel does not match individual processing"
    assert np.allclose(result[:, :, 2], result_b), \
        "B channel does not match individual processing"


def test_median_filter_rgb_invalid():
    """Test: Error when image is not RGB."""
    # 2-channel image
    image_2channels = np.ones((5, 5, 2))

    with pytest.raises(ValueError, match="Image must have 3 channels"):
        apply_median_filter_rgb(image_2channels)

    # 4-channel image
    image_4channels = np.ones((5, 5, 4))

    with pytest.raises(ValueError, match="Image must have 3 channels"):
        apply_median_filter_rgb(image_4channels)


def test_median_filter_edges():
    """Test: Filter behavior at edges."""
    matrix = create_test_matrix(7, 'edges')

    # Apply filter
    result = apply_median_filter(matrix, kernel_size=3)

    # Verify that edges are preserved (median filter with reflect padding)
    # Corner (0,0): reflected neighborhood will have high values (edges)
    manual_corner = compute_manual_median(matrix, 0, 0, kernel_size=3)
    filtered_corner = result[0, 0]

    assert np.isclose(manual_corner, filtered_corner, rtol=1e-7), \
        f"Corner (0,0): manual={manual_corner}, filtered={filtered_corner}"


def test_compute_manual_median():
    """Test: Manual median computation."""
    matrix = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ], dtype=np.float32)

    # Center position (2,2) with 3x3 kernel
    # Neighborhood: [[7,8,9],[12,13,14],[17,18,19]]
    # Values: [7,8,9,12,13,14,17,18,19], Median = 13
    manual = compute_manual_median(matrix, 2, 2, kernel_size=3)
    assert manual == 13.0, f"Manual median expected 13.0, got {manual}"

    # Corner (0,0) with 3x3 kernel (with reflect padding)
    # Reflected neighborhood: [[1,1,2],[1,1,2],[6,6,7]]
    # Values sorted: [1,1,1,1,2,2,6,6,7], Median = 2
    manual_corner = compute_manual_median(matrix, 0, 0, kernel_size=3)
    assert manual_corner == 2.0, f"Corner (0,0) expected 2.0, got {manual_corner}"


def test_verify_median_filter():
    """Test: Complete verification function."""
    matrix = create_test_matrix(6, 'random')

    # Verify all positions
    all_correct, results = verify_median_filter(
        matrix, kernel_size=3, positions=None
    )

    assert all_correct, "Not all positions match"

    # Verify some specific positions
    positions = [(0, 0), (3, 3), (5, 5)]
    all_correct, results = verify_median_filter(
        matrix, kernel_size=3, positions=positions
    )

    assert all_correct, f"Specific positions do not match: {results}"


def test_process_npy_images():
    """Test: Processing .npy files."""
    # Create temporary folder for tests
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")

        os.makedirs(input_dir)

        # Create some test images
        for i in range(3):
            # 10x10 RGB image
            image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
            path = os.path.join(input_dir, f"image_{i}.npy")
            np.save(path, image)

        # Process images
        process_npy_images(input_dir, output_dir, kernel_size=3, process_rgb=True)

        # Verify that output files were created
        output_files = os.listdir(output_dir)
        assert len(output_files) == 3, \
            f"Expected 3 output files, found {len(output_files)}"

        # Verify that files contain filtered images
        for filename in output_files:
            path = os.path.join(output_dir, filename)
            filtered_image = np.load(path)
            assert filtered_image.shape == (10, 10, 3), \
                f"Image {filename} does not have the expected shape"


def test_process_npy_images_errors():
    """Test: Error handling in processing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Input folder does not exist
        with pytest.raises(ValueError, match="Input folder does not exist"):
            process_npy_images(
                "/nonexistent/path",
                os.path.join(temp_dir, "output"),
                kernel_size=3
            )

        # Create input folder
        input_dir = os.path.join(temp_dir, "input")
        os.makedirs(input_dir)

        # Invalid kernel_size
        with pytest.raises(ValueError, match="kernel_size must be a positive odd number"):
            process_npy_images(input_dir, temp_dir, kernel_size=4)


def test_create_test_matrix():
    """Test: Test matrix creation."""
    # Ordered matrix
    ordered = create_test_matrix(3, 'ordered')
    expected = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.float32)
    assert np.array_equal(ordered, expected), "Ordered matrix incorrect"

    # Random matrix (same seed)
    np.random.seed(42)
    random1 = create_test_matrix(3, 'random')
    np.random.seed(42)
    random2 = create_test_matrix(3, 'random')
    assert np.array_equal(random1, random2), "Random matrix not reproducible"

    # Stepped matrix
    stepped = create_test_matrix(3, 'stepped')
    expected_stepped = np.array([[0, 1, 2], [1, 1, 2], [2, 2, 2]], dtype=np.float32)
    assert np.array_equal(stepped, expected_stepped), "Stepped matrix incorrect"

    # Invalid type
    with pytest.raises(ValueError, match="Invalid matrix type"):
        create_test_matrix(3, 'invalid')

def test_compute_noi_identical():
    """Test: Identical distributions should yield NOI = 1."""
    bins = 40
    range_min, range_max = 0.0, 100.0

    # Create identical histograms
    hist1 = np.ones(bins) * 100
    hist2 = hist1.copy()

    noi = compute_noi(hist1, hist2, (range_min, range_max), bins=bins)

    assert np.isclose(noi, 1.0, rtol=1e-7), \
        f"Identical distributions should give NOI=1, but got {noi}"


def test_compute_noi_no_overlap():
    """Test: Non-overlapping distributions should yield NOI ~ 0."""
    bins = 40
    range_min, range_max = 0.0, 100.0

    # Non-overlapping histograms
    hist1 = np.zeros(bins)
    hist2 = np.zeros(bins)
    hist1[:10] = 100   # Only first 10 bins
    hist2[30:] = 100   # Only last 10 bins

    noi = compute_noi(hist1, hist2, (range_min, range_max), bins=bins)

    assert noi < 0.01, \
        f"Non-overlapping distributions should give NOI~0, but got {noi}"


def test_compute_noi_empty_histogram():
    """Test: Empty histogram should yield NOI = 0."""
    bins = 40
    range_min, range_max = 0.0, 100.0

    hist1 = np.ones(bins) * 100
    hist2 = np.zeros(bins)  # Empty histogram

    noi = compute_noi(hist1, hist2, (range_min, range_max), bins=bins)

    assert noi == 0.0, \
        f"Empty histogram should give NOI=0, but got {noi}"


def test_compute_noi_normalization():
    """Test: Verify that histograms are normalized correctly."""
    bins = 40
    range_min, range_max = 0.0, 100.0

    # Histograms with different areas
    hist1 = np.ones(bins) * 100  # Area = 100 * 40 = 4000
    hist2 = np.ones(bins) * 50   # Area = 50 * 40 = 2000

    noi = compute_noi(hist1, hist2, (range_min, range_max), bins=bins)

    # Should give NOI = 1 after normalization
    assert np.isclose(noi, 1.0, rtol=1e-7), \
        f"Proportional histograms should give NOI=1, but got {noi}"


def test_simulate_distributions():
    """Test: Verify that simulation generates consistent results."""
    np.random.seed(42)  # For reproducibility

    mu1, sigma1 = 30.0, 5.0
    mu2, sigma2 = 50.0, 7.0

    hist1, hist2, range_values, simulated_noi = simulate_distributions(
        mu1, sigma1, mu2, sigma2, bins=40, n_samples=10000
    )

    # Verify shapes
    assert hist1.shape == (40,), f"hist1 shape: {hist1.shape}"
    assert hist2.shape == (40,), f"hist2 shape: {hist2.shape}"

    # Verify that NOI is in [0, 1]
    assert 0.0 <= simulated_noi <= 1.0, \
        f"NOI should be between 0 and 1, but got {simulated_noi}"

    # Verify ranges
    range_min, range_max = range_values
    assert range_min == 0.0
    assert range_max == 100.0


def test_analytical_noi_gaussians():
    """Test: Compare analytical NOI with expected values."""
    # Case 1: Identical distributions
    noi1 = analytical_noi_gaussians(50.0, 10.0, 50.0, 10.0)
    assert np.isclose(noi1, 1.0, rtol=1e-5), \
        f"Identical distributions: expected 1.0, got {noi1}"

    # Case 2: Very separated distributions
    noi2 = analytical_noi_gaussians(20.0, 2.0, 80.0, 2.0)
    assert noi2 < 0.001, \
        f"Separated distributions: expected <0.001, got {noi2}"

    # Case 3: Distributions with moderate overlap
    noi3 = analytical_noi_gaussians(40.0, 10.0, 60.0, 10.0)
    assert 0.1 < noi3 < 0.5, \
        f"Moderate overlap: expected between 0.1 and 0.5, got {noi3}"


def test_consistency_simulation_analytical():
    """Test: Compare simulation with analytical computation."""
    np.random.seed(123)

    mu1, sigma1 = 35.0, 8.0
    mu2, sigma2 = 45.0, 9.0

    # Simulation
    hist1, hist2, range_values, simulated_noi = simulate_distributions(
        mu1, sigma1, mu2, sigma2, bins=100, n_samples=50000
    )

    # Analytical
    analytical_noi = analytical_noi_gaussians(
        mu1, sigma1, mu2, sigma2, range_min=0.0, range_max=100.0, n_points=10000
    )

    # They should be similar (5% tolerance)
    relative_difference = abs(simulated_noi - analytical_noi) / analytical_noi

    assert relative_difference < 0.05, \
        f"Difference too large: simulated={simulated_noi}, " \
        f"analytical={analytical_noi}, difference={relative_difference * 100:.2f}%"


def test_edge_values():
    """Test: Behavior in edge cases."""
    bins = 40
    range_min, range_max = 0.0, 100.0

    # Case: single non-empty bin in each histogram, no overlap
    hist1 = np.zeros(bins)
    hist2 = np.zeros(bins)
    hist1[0] = 100    # Only first bin
    hist2[-1] = 100   # Only last bin

    noi = compute_noi(hist1, hist2, (range_min, range_max), bins=bins)

    # NOI should be 0 (no bin overlap)
    assert noi == 0.0, f"Opposite edges should give NOI=0, but got {noi}"

    # Case: all bins have the same value
    hist3 = np.full(bins, 50.0)
    hist4 = np.full(bins, 50.0)
    noi2 = compute_noi(hist3, hist4, (range_min, range_max), bins=bins)
    assert np.isclose(noi2, 1.0, rtol=1e-7), \
        f"Uniform distributions should give NOI=1, but got {noi2}"

print("All test functions defined.")
