<img src="https://github.com/cuongth95/gigantic_dataset/blob/main/gida.png" width="128" alt="gida logo made by ChatGPT">
---
This is the official repository for the Gigantic Dataset (Gida).

## Prerequisite - How to create a config

Download all WDNs from Drive and put them into the folder `gigantic-dataset/inputs/public`

Call `check_wdn_and_collect_stats()` to generate the `profiler_report.json` storing stats of all WDNs

Call `create_blueprint_config(<input_path>,<new_yaml_path>) to create a <blueprint_config>.yaml

 Direct to `gigantic-dataset/arguments` and copy the `<blueprint_config>.yaml`
 
Rename the created yaml file. Prefered naming format is `<wdn_name>_<something>.yaml`
The structure of the yaml file is:
```

        |
        |_general_field_1: value_1
        |_general_field_2: value_2
        |_<component>_tune: 
            |_<param>_strategy:
            |_<param>_values:

```
### Manual optimization tutorial
In this tutorial, we focus on tuning param strategy and values as follows:
1. Beginning with a pure blueprint config file, run the simulation and check the number of successful runs.
	- Use command `simulate_report(yaml_path=r'<blueprint_config>.yaml')` to run the simulation
	- If (success/ expected runs) >= 50%, move to step 2.
	- Otherwise:
		- Open EPANET, run the simulation, and check the negative pressure at nodes. If there exist negative nodes, put their names into the `skip_names` list in the `<blueprint_config>.yaml`
  		- Re-run and check #success. 
		- If it still has zero success case, manually tune `junction_elevation`, `reservoir_base_head` in `<blueprint_config>.yaml` until the (success/ expected) >=50%
2. Pick the strategy. `<param>_strategy`  and assign its corresponding required parameters ( Refer to the parameters and suggested values in the next Section). There are 9 strategies, as follows:
 - **sampling**: random in the range [min, max]
 - **perturbation**: Gaussian sampling
 - **adg**: Andres' Demand generator for demand only
 - **adg_v2**: Andres' Demand generator V2 for demand only
 - **series**: load an existing series into the target value list
 -  **keep**: reuse the target value loaded from INP.
 - **factor**: linear transform where scale and bias are randomly created
 - **substitute**: pick a value from baseline, add noise, then share it with others
- **terrain**: for `junc_elevation` only. Create terrain and map height to nodes.   
3. After tuning a parameter, run the simulation in 100 runs, and check the success runs. To run, use `simulate_report(yaml_path=r'<created_config_name>.yaml')`
  	1. If (success/ total) in [0.4, 0.6]:
		   We successfully tuned the parameter! Pick the next param, and repeat Step 2.
	           The order of choosing parameters can be the same as the one in the Excel or JSON file.
 	2. Otherwise, after few trails, retry:
		1. Change the strategy from 'sampling' to 'perturbation'
		2. Pick std following this formula: `new_std = std / 4**i`, where `i` is the current trial time
	3. In tough cases, `keep-null` and move to another parameter.

## Strategy and parameters
We discuss the strategy parameter, its meaning, and suggested values.
### Sampling
It is a uniform sampling in the range [min, max]. 
1. If the param is not curve-related, `values` is a list of 2 elements: [min, max] (order-insensitive):

```
        |_junction_tune: 
                |_elevation_strategy: sampling
                |_elevation_values:
                | - 1   #min
                | - 100 #max
```
2. If the param is curve-related, `values` is a list of 5 elements: [xmin, xmax, ymin, ymax, num_points] (order-sensitive):

``` 
        |_pump_curve_name_strategy: sampling
            |_pump_curve_name_values: 
            | - 0       #xmin
            | - 0       #xmax
            | - 0.036   #ymin
            | - 0.09    #ymax
            | - 12      #num_points_in_curve
``` 
          
For recommendation, please refer to `profiler.json` and pick the `Q1` and `Q3` of the target parameter as `min` and `max` respectively.

### Perturbation
It is Gaussian sampling N(mean, std). It applies to every component and, therefore, each has its own value. Thus, it is insufficient for parameters requiring consistency (e.g. pipe diameter) throughout a scenario.
1. If the param is not curve-related, `values` is a list of 1 element: [std]:
``` 
        |_junction_tune: 
                |_elevation_strategy: perturbation
                |_elevation_values:
                | - 1.45   #std
```
2. If the param is curve-related, `values` is a list of 2 elements: [xstd, ystd]:
``` 
         |_pump_curve_name_strategy: perturbation
            |_pump_curve_name_values: 
            | - 0.2     #xstd
            | - 12.     #ystd
```

For recommendation, please refer to `profiler.json` and pick the `std`  of the target parameter,

### ADG (v1)
It only supports `junction_base_demand` (input demand), values is a list of 3 elements: [seasonal, frequency, scale]:
```
        |_junction_tune: 
                |_base_demand_strategy: adg
                |_base_demand_values:
                | - 1       #seasonal 0 or 1
                | - 365     #frequency
                | - 0.002   #target value = <scale> * normed_value generated by ADG #check min max of junc_demand to choose the right scale
```
We always fix the `seasonal` and `frequency`, while `scale` is set to `Q3` of `junction_base_demand` referred from `profiler.json`.

### ADG v2
It only supports `junction_base_demand` (input demand).
ADGv2 will generate multipliers whose range strictly is in [0,1]. Then, `scale` plays a role in scaling them to a realistic amount. Note that `scale` here is the maximum scale that can be achieved. The actual scale could be smaller or equal `scale`. 
At the end of a blueprint config, there are static, scenario-shared parameters for adg_v2, they are more sophisticated if you prefer a stronger customization.
```
        |_junction_tune: 
                |_base_demand_strategy: adg_v2
                |_base_demand_values:
                | - 0.002   #scale
```
`scale` is set to `Q3` of `junction_base_demand` referred from `profiler.json`.

(UPDATE LATER)

### Series
Applicable for pattern-related parameters! It loads an existing time series and parses it to a pattern parameter. Note that the series is **scenario-shared**.
Values is a list of T value, where T is the number of time steps (time duration)
```
        |_junction_tune: 
                |_base_demand_strategy: series
                |_base_demand_values:
                | - 0.1      # t= 0
                | - 0.12     # t= 1
                | - ...      
                | - 0.05     # t= T-1  
```
No recommendation since it is externally loaded.
### Keep
 Used by default for all parameters.  If a value exists from the baseline, we re-use and fix it across scenarios.
Some components often miss values. Inspired by EPANET, we impute them with the default value gathered from the local baseline or the value from other networks in `profiler.json`.
If none of the baseline networks has a non-zero value, we skip this parameter.
```
        |_<component>_tune: 
                |_<param>_strategy: keep
                |_<param>_values: null
```
### Factor
Indeed, it is a linear transformation. Given a particular parameter `x`,  we compute this formula: `new_x = scale * x + bias`, where `scale` and `bias` are the maximum scale, and maximum bias could be achieved.  In addition, both are internally shared among components within the same scenario.
It ensures **consistency** and is less extreme than in the `perturbation` strategy.
Note: This strategy is useful for `pipe_diameter`. It is encouraged not to apply this strategy to pattern and curve parameters since the testing process has not been done.
```
        |_<component>_tune: 
                |_<param>_strategy: factor
                |_<param>_values: 
                |       - 0.  # min scale
                |	- 1.0 # max scale
                |	- 0.5 # bias
```
For recommendation, proper `scale` should be set in a way that does not cause the affected value to exceed the `max` limit. `bias` can be set as `std` of this parameter  in `profiler.json`.

### Substitute
Similar to Factor, it retrieves a value and uses it for all components in the same scenario. However, `substitute` randomly takes a value of one of the components in the baseline network. It also  offers a `bias` to provide a minor difference between two considering components.
This strategy ensures **consistency**.
```
        |_<component>_tune: 
                |_<param>_strategy: substitute
                |_<param>_values: 
                |	- 0.5 # bias
```
For recommendation, `bias` can be set as `std` of this parameter  in `profiler.json`.

### Terrain
This strategy is applicable for `junction_elevation`. In particular, it creates a terrain (2D height map) using the diamond-square algorithm. Then, based on the junctions' coordinates, we project the network onto the map and retrieve the new elevation value. 

Note: It is applicable to even networks that miss elevation values. In such a case, we dump the coordinates from [nx.spring_layout](https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html).
```
        |_junction_tune: 
                |_elevation_strategy: terrain
                |_elevation_values: 
                |	- 10.2 	# noise
                |	- 17 	# terrain width
```
For recommendation,  the `noise` should be set in the range [4, 25]. Experimental tests show that the range offers flat-like and hill-like terrains. Notably, the higher the noise is, the more steep the terrain is.
You could fix the `terrain width` to `17` since it seems less effective to the simulation quality. The only requirement is the number be odd.
