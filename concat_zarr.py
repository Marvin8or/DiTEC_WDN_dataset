import os
from ditec_wdn_dataset.utils.auxil_v8 import concatenate_zarr

# Hardcoded list of zarr array paths to concatenate
zarr_paths = [
    # "outputs_test/simgen_Net2_20250714_1518.zarr",
    # "outputs_test/simgen_Net2_20250714_1520.zarr",
    # "outputs_test/simgen_Net2_20250714_1522.zarr",
    # "outputs_test/simgen_Net2_20250714_1524.zarr",
    # "outputs_test/simgen_Net2_20250714_1526.zarr",
    # "outputs_test/simgen_Net2_20250714_1528.zarr",
    "outputs_test/simgen_Net3_20250714_1540.zarr",
    "outputs_test/simgen_Net3_20250714_1542.zarr",
    "outputs_test/simgen_Net3_20250714_1543.zarr",
    "outputs_test/simgen_Net3_20250714_1545.zarr",
    "outputs_test/simgen_Net3_20250714_1547.zarr",
    "outputs_test/simgen_Net3_20250714_1548.zarr",
    "outputs_test/simgen_Net3_20250714_1550.zarr",
    "outputs_test/simgen_Net3_20250714_1552.zarr",
    "outputs_test/simgen_Net3_20250714_1554.zarr",
    "outputs_test/simgen_Net3_20250714_1556.zarr",
    "outputs_test/simgen_Net3_20250714_1558.zarr",
    "outputs_test/simgen_Net3_20250714_1600.zarr",
    "outputs_test/simgen_Net3_20250714_1603.zarr",
    "outputs_test/simgen_Net3_20250714_1605.zarr",
]

# Define the output path for the concatenated zarr array
output_zarr_path = (
    "outputs_test/Net3_test.zarr"  # Change this variable as needed
)

if os.path.exists(output_zarr_path):
    raise FileExistsError(f"Output path already exists: {output_zarr_path}")

print(f"Concatenating {len(zarr_paths)} zarr arrays into: {output_zarr_path}")
concatenate_zarr(zarr_paths, output_zarr_path, verbose=True)
print(f"Done. Concatenated zarr saved to: {output_zarr_path}")
