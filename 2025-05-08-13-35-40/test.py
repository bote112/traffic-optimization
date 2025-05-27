import os
import sys
import simulation_generator as sim_gen

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise EnvironmentError("Please set the SUMO_HOME environment variable.")

import traci

sumo_cmd = ["sumo", "-c", "E:/licenta/oras-mic/2025-05-08-13-35-40/osm.sumocfg"]

traci.start(sumo_cmd)


while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()


traci.close()
