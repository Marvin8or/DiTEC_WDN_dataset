# Parameters
An input attribute is named as `<component>_<attribute>`. Nodal components include `reservoir`, `junction`, and `tank`, while edge components involve `pipe`, `headpump`, `powerpump`, etc.

Tip: Open `.zip` file to see available attributes as filename (csv) or folder name (zarr).

On the other hand, another kind of attribute is simulation output that has no component prefix (e.g., velocity, pressure, ...). They concatenate features of components based on their type (node or link).Therefore, we might encounter a mismatch in size when striving to stack input and output parameters. Consider this example:
```python
# This should raise an error
GidaV6(
    zip_file_paths=[
        r"./Dataset/simgen_Anytown_20240524_1202_csvdir_20240527_1205.zip",  # Anytown datset
    ],
    node_attrs=[
        "junction_base_demand",                                         # load junc base_demand (#junctions)
       ("reservoir_base_head", "junction_elevation", "tank_elevation"), # load node elevation(#reservoirs + #tanks + #junctions)             
    ],  
    num_records=100,  # take only 100 records
)
```
Intuitively, we can observe the size inconsistency between `junction_base_demand` and the tuple of elevation-related parameters. However, we sometimes want to define `node_attrs` in this way.\
To solve this, GiDA offers the `*` operator indicating a specific parameter whose size is less than others. Let's fix the above example:
```python
GidaV6(
    zip_file_paths=[
         r"./Dataset/simgen_Anytown_20240524_1202_csvdir_20240527_1205.zip"  # Anytown datset
    ],
    node_attrs=[
        "*junction_base_demand",                                         # load junc base_demand (#junctions) with asterisk
       ("reservoir_base_head", "junction_elevation", "tank_elevation"), # load node elevation(#reservoirs + #tanks + #junctions)             
    ],  
    num_records=100,  # take only 100 records
)
```
In this way, GiDA pads the incomplete parameters according to the tuple or non-asterisk parameters.
