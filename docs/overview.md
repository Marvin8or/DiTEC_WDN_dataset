# GiDA - The Gigantic Dataset

This work includes a collection of synthetic scenarios devised from 36 **Water Distribution Networks (WDNs)**. 

For the sake of clarity, it would be better to get into familiarized concepts:

* **Scenario** denotes as a sequence of snapshots.

* **Snapshot** represents a measured steady-state of a particular WDN and is often modelled as an undirect graph.

* **Input parameters** includes simulation inputs, such as demands, pipe diameter, and so on.

* **Output parameters** includes simulation outcomes which researchers are interested in (e.g., pressure, flow rate, head, ...)

Both parameters are described as nodal/edge features in the snapshot graph. Their values are diverse but temporal correlated with those of other snapshots in the **same** scenario. 
However, in GiDA, two scenarios are considered completely different WDNs despite their origin being the same network.



# Acknowledgement
This work is funded by the project DiTEC: Digital Twin for Evolutionary Changes in Water Networks (NWO 19454).

# Citing GiDA

* For the up-to-date dataset and interface, please use this:
```
TODO: UPDATE LATER
```

* For the older dataset versions, please use this:
```tex
@article{tello2024largescale,
    AUTHOR = {Tello, Andrés and Truong, Huy and Lazovik, Alexander and Degeler, Victoria},
    TITLE = {Large-Scale Multipurpose Benchmark Datasets for Assessing Data-Driven Deep Learning Approaches for Water Distribution Networks},
    JOURNAL = {Engineering Proceedings},
    VOLUME = {69},
    YEAR = {2024},
    NUMBER = {1},
    ARTICLE-NUMBER = {50},
    URL = {https://www.mdpi.com/2673-4591/69/1/50},
    ISSN = {2673-4591},
    DOI = {10.3390/engproc2024069050}
}
```
