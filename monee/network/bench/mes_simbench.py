from monee.io.from_simbench import obtain_simbench_net

from ..mes import generate_mes_based_on_power_net


def generate_mes_based_on_simbench_id(
    simbench_id: str,
    heat_deployment_rate,
    gas_deployment_rate,
    chp_density=0.1,
    p2g_density=0.02,
    p2h_density=0.1,
):
    return generate_mes_based_on_power_net(
        obtain_simbench_net(simbench_id),
        heat_deployment_rate,
        gas_deployment_rate,
        chp_density=chp_density,
        p2g_density=p2g_density,
        p2h_density=p2h_density,
    )
