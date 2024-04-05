# All the files used in WI simulation.
Antennas folder:
  * It has the antenna patterns we used for WI simulations.
  * We adapted them from Talon measurements \cite{}
  * The way WI accepts user defined antenna patterns are different from what Talon provides.
  * We need to change i) the way we read the numbers, ii) map the floating point angle increments to the closest integer angle increments

Automation folder:
  * Python scripts to automate simulation runs.
  * There are two: i) Automatically place Tx and 200 Rx, and ii) Automatically try 34 Tx beams for each 200 Rx points

Codes folder:
  * Python scripts to i) generate Tx and 200 Rx coordinates, ii) Combine individual 200 result files into one file, and iii) Initial evaluation codes
