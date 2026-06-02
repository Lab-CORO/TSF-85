# GUI Guide — TSF-85 Tactile Sensor Extension

This guide describes how to use the TSF-85 extension through the Isaac Sim GUI panel.

## Opening the panel
Launch Isaac Sim with the extension enabled, then open the **TSF-85 Tactile Sensor**
panel from the Isaac Sim window menu.

## Workflow

1. Select a primitive in the scene that contains the sensor.
2. The extension identifies the soft object corresponding to the dielectric for file saving.
3. Change the output directory. Default location: `/home/User/Documents`.
4. Rename the generated files as needed. Default file name: `TactileData`.
5. In the "Tactile map visualization" section, select the checkbox to enable real-time
   viewing of the generated tactile maps.
6. Start the simulation.
7. Stop the simulation to end data saving.

## Notes
* Data is recorded while the timeline is playing and the files are closed cleanly when
  you press Stop.
* The output file names are derived from the base name you set (see the
  [main README](../README.md) for the full list of generated files).
