# DiTEC-WDN - The Gigantic Dataset

This work includes a collection of synthetic scenarios devised from 36 **Water Distribution Networks (WDNs)**. 

For the sake of clarity, it would be better to get into familiarized concepts:

* **Scenario** denotes as a sequence of snapshots.

* **Snapshot** represents a measured steady-state of a particular WDN and is often modelled as an undirect graph.

* **Input parameters** includes simulation inputs, such as demands, pipe diameter, and so on.

* **Output parameters** includes simulation outcomes which researchers are interested in (e.g., pressure, flow rate, head, ...)

Both parameters are described as nodal/edge features in the snapshot graph. Their values are diverse but temporal correlated with those of other snapshots in the **same** scenario. 
However, in DiTEC-WDN, two scenarios are considered completely different WDNs despite their origin being the same network.


# Acknowledgement
This work is funded by the project DiTEC: Digital Twin for Evolutionary Changes in Water Networks (NWO 19454).

# Citing DiTEC-WDN

If you use the dataset, please cite:

```latex

@misc{huy2025dwd}{
    title={DiTEC-WDN: A Large-Scale Dataset of Water Distribution Network Scenarios under Diverse Hydraulic Conditions}, 
    author={Huy Truong and Andr\'{e}s Tello and Alexander Lazovik and Victoria Degeler},
    year={2025},
    note = {HT and AT contributed equally to this work. The dataset is linked to a paper submitted to *Nature Scientific Data*.}
}


```