# TFG_JoanCamps
Several software resources used during the development of the project.

**Abstract:** 

The project adopts a usual paradigm of scientific inquiry: formulating simplified theoretical models to study complex phenomena and ultimately translating those insights back to reality. We introduce two mathematically distinct slow-fast systems—one smooth and one piecewise linear (PWL)—under the FitzHugh-Nagumo framework to reproduce the dynamics of a simple circuit based on a non-linear resistor. Phenomena such as the canard explosion, antimonotonicity, and bursting driven by a slow passage through a bifurcation are formalized to identify the differences and limitations of the models. Notably, while both models generally capture these complex dynamics, the PWL model must be modified to exhibit the delay effect that occurs naturally in the smooth system during specific bursting scenarios. The project concludes by validating these phenomena through computer simulations and an analogue implementation of the circuit. Experimental results show that the circuit simulation using SPICE software fairly aligns with the models' predictions, while the physical analogue results reveal these phenomena within an unexpected range of parameter configurations.

**Folder Structure**
- GeoGebra > Files containing representations of the nullclines and slow manifolds of some PWL models.
- Maxima > Computations for Theorem 14 and for the discussion in PWL's delay phenomenon.
- LTSpcie > Several schematics, with and without forcing, for obtaining circuit's behaviour and nullclines.
- Python > Several scripts involving bifurcation diagrams and plots for the theoretical models and SPICE simulations

