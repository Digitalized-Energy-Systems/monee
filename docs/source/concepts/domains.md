
# Physical model

Currently we only provide a rough overview of the physical model per default used by monee.

### Electricity (AC steady-state)

For the electricity model implemented in monee, the well-known AC steady state equations are used.

### Gas (steady-state)

The pressure drops are calculated using the Weymouth equation. For the friction we use Swamee-Jain.

### Water (steady-state)

The pressure drops are calcualted using the Darcy-weisbach equation. For the friction we use Swamee-Jain.

The temperature loss is calculated by applying the the heat transfer equations through cylindric pipe-walls, considering insulation.
