
# Download datasets
TODO: UPDATE LATER.
##  GiDA-V1
Please go to [Gida-V1](https://zenodo.org/records/11353195), download the dataset, and place it into a folder, say `/Dataset`.


# Tutorial
For the first-time user, please refer to the `datasets.py` script and review the `GidaV6.__init__` function. A minimal example is also provided at the end of the script.

The data interface `GidaV6` will take node (edge) attributes and output a set of records. Each records is a `Data` instance (visit [here](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data) for more information). This `Data` contains a snapshot graph described by the (sparsed) adjacency matrix A, nodal feature X, and edge feature E. Also, if label is available, we have label Y corresponding to either node or edge. In the case both edge and node sets have their own labels, Y is for label of nodes, while E_Y stands for label of edges.

Assume you want to load the train set of Anytown network, a very simple interface can be declared as follow:
```
from gigantic_dataset.core.datasets import GidaV6
gida = GidaV6(
            zip_file_paths=[
                r"./Dataset/simgen_Anytown_20240524_1202_csvdir_20240527_1205.zip",    # Anytown datset
            ],
            node_attrs=[
                "demand",                                                             # load nodal demand
            ],                                               
            edge_attrs=["pipe_diameter", "pipe_length"],                              # load some properites at edge
            label_attrs=["pressure"],                                                 # expect labels Y are pressure
            edge_label_attrs=["flowrate"],                                            # expect edge labels E_Y are flowrate
            split_set="train",                                                        # take train set only
            num_records=100,                                                          # take only 100 records
            selected_snapshots=None,                                                  # take all snapshots
        )
# You can call a record directly
print(gida[0]) # Data instance
# Or via a data loader
from torch_geometric.loader import DataLoader
loader = DataLoader(gida, batch_size=1)
print(next(iter(loader))) #Batch instance
```
