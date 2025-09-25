
# Quickstart

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

## Next steps

- See the [tutorials](tutorials/index) for end-to-end workflows.
- Explore the [API Reference](api/index).
