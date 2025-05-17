### SUMO REFERENCES ###

# SUMO #

- [Main sumo documentation page](https://sumo.dlr.de/docs/index.html)
- [Tutorials](https://sumo.dlr.de/docs/Tutorials/index.html)
- [SUMO repo](https://github.com/eclipse-sumo/sumo/tree/main)
- [Downloads](https://sumo.dlr.de/docs/Downloads.php)

Main tools : sumo-gui, netedit

# TraCI : Traffic Control Interface #

- [Introduction](https://sumo.dlr.de/docs/TraCI.html)
- [Documentation page](https://sumo.dlr.de/pydoc/traci.html)
- [Traci Code](https://github.com/eclipse-sumo/sumo/tree/main/tools/traci)

Some tutorials : 

- [Traffic Lights](https://sumo.dlr.de/docs/Tutorials/TraCI4Traffic_Lights.html)
- [ParkingLot responsive parking service](https://sumo.dlr.de/docs/Tutorials/CityMobil.html)
- [Pedestrian Actuated Crossing](https://sumo.dlr.de/docs/Tutorials/TraCIPedCrossing.html)

# Sumolib : tools to work with sumo networks #

- [Quickstart](https://sumo.dlr.de/docs/Tools/Sumolib.html)
- [Documentation Page](https://sumo.dlr.de/pydoc/sumolib.html)
- [Sumolib Code](https://github.com/eclipse-sumo/sumo/tree/main/tools/sumolib)

To import TraCI and Sumolib in python script : 
```
import os, sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:   
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
import sumolib
```