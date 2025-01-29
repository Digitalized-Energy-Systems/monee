# monee
[PyPi](https://pypi.org/project/monee/) | [Docs](https://monee.readthedocs.io)


![lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/Digitalized-Energy-Systems/monee/blob/development/LICENSE)
[![Test mango-python](https://github.com/Digitalized-Energy-Systems/monee/actions/workflows/test-monee.yml/badge.svg)](https://github.com/Digitalized-Energy-Systems/monee/actions/workflows/test-monee.yml)
[![codecov](https://codecov.io/gh/Digitalized-Energy-Systems/monee/graph/badge.svg?token=KSBSBQGNBZ)](https://codecov.io/gh/Digitalized-Energy-Systems/monee)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Digitalized-Energy-Systems_monee&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=Digitalized-Energy-Systems_monee)

The python project `monee` (a Modular Network-based Energy Grid Optimization) can be used to calculate the steady-state energy flow of coupled grids (electricity, water (heating), gas). It is also capable to flexibly formulate and solve optimal energy flow problems. For this `monee` currently uses [GEKKO](https://gekko.readthedocs.io/en/latest/) to solve these problems (further integrations are planned).

Further, there are unique key aspects of monee such as:
* Timeseries simulation
* Native support of multi-energy components as P2H/CHP/P2G
* Integration of networkx as main internal datastructure, this enables easy appliance of graph-based approaches.
* Modular component definitions
* Importing of MATPOWER networks
* Restricted import of [pandapower](pandapower.org) and therefore [simbench](simbench.de) networks.

# Installation

The `monee` framework is hosted on pypi, as such you can install it with:

```bash
pip install monee
```

# Examples

## Creating a network (express API)

```python
from monee import mx, run_energy_flow

net = mx.create_multi_energy_network()
bus_0 = mx.create_bus(net)
bus_1 = mx.create_bus(net)

mx.create_line(bus_0, bus_1, 100, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007)
mx.create_ext_power_grid(bus_0)
mx.create_power_load(bus_1, 0.1)

result = run_energy_flow(net)
```
