This repository contains Python code for performing the model predictive control simulations from Abel, Chakrabarty, and Doyle, IFAC World Congress 2017.

Contents:
Each script performs simulations and genera . Note that a "sim=False" boolean is defined before each simulation. If set to true, the simulation will be performed. If set to false, data will be taken from the "Data" folder to generate the figures.

All final figures used in the manuscript are included in the "Figures" directory, and were generated using the data.

Python 2.7
Dependencies:
- numpy
- scipy
- matplotlib
- CasADi
- pyswarm
- futures concurrency


Abstract:
Recent in vitro studies have identified small-molecule pharmaceuticals effecting dosedependent changes in the mammalian circadian clock, providing a novel avenue for control. Most studies employ light for clock control, however, pharmaceuticals are advantageous for clock manipulation through reduced invasiveness. In this paper, we employ a mechanistic model to predict the phase dynamics of the mammalian circadian oscillator under the effect of the pharmaceutical under investigation. These predictions are used to inform a constrained model predictive controller (MPC) to compute appropriate dosing for clock re-entrainment. Constraints in the formulation of the MPC problem arise from variation in the phase response curves (PRCs) describing drug effects, and are in many cases non-intuitive owing to the nonlinearity of oscillator phase response effects. We demonstrate through in-silico experiments that it is imperative to tune the MPC parameters based on the drug-specific PRC for optimal phase manipulation.


