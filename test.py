import os
import shutil
import zarr
from ditec_wdn_dataset.utils.auxil_v8 import concatenate_zarr

def test_concatenate_zarr():
    # Example hardcoded zarr paths (ensure these exist for the test)
    zarr_paths = [
        'outputs/simgen_Net1_20250702_1411.zarr',
        'outputs/simgen_Net1_20250702_1413.zarr',
    ]
    concat_path = 'outputs/test_concat.zarr'
    # Remove if exists from previous test
    if os.path.exists(concat_path):
        shutil.rmtree(concat_path)
    # Run concatenation
    concatenate_zarr(zarr_paths, concat_path, verbose=False)
    # Check file exists
    assert os.path.exists(concat_path), f"Concatenated zarr file not created: {concat_path}"
    # Check shapes
    g0 = zarr.open_group(zarr_paths[0], mode='r')
    g1 = zarr.open_group(zarr_paths[1], mode='r')
    gcat = zarr.open_group(concat_path, mode='r')
    for arr_name in g0.array_keys():
        arr0 = g0[arr_name]
        arr1 = g1[arr_name]
        arrcat = gcat[arr_name]
        assert arrcat.shape[0] == arr0.shape[0] + arr1.shape[0], f"Shape mismatch for {arr_name}"
        assert arrcat.shape[1:] == arr0.shape[1:], f"Shape mismatch for {arr_name}"
    # Cleanup
    shutil.rmtree(concat_path)
    print("test_concatenate_zarr passed.")

if __name__ == "__main__":
    test_concatenate_zarr()
