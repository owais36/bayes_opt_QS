# QuadSAT radiation pattern

Three maxima search methods are implemented in the provided code:

1) Circular search (closest analogy would be depth first search where the drone flies in circles and take the maxima at circumference as the center of the next circle. The process is stopped when the maxima on circumference is lower than the center of the circle)

2) Gradient Ascent (Starting at a random point the drone makes a small move in azimuth and elevation. A second order curve is fitted to the recorded signal values and the derivative is calculated at the point. Using standard gradient ascent formula and learning rate alpha we select the next point. The process is stopped when the value of derivative is very close to zero indicating that the maxima is achieved)

3) Concurrent cuts (In this method the drone flies an azimuth cut and then an elevation cut. The location of maxima in azimuth and in elevation together give a good approximate of the center of the main beam.)

**Bayesian Black Box Model Approach**

In order to get the maximum information about radiation pattern with as few signal samples as possible we plan to use Bayesian optimization process. The steps in Bayesian Optimization are stated below:

1) Assumes a surrogate model which is usually gaussian and get its prior probabilities
2) Given set of observation(s), update the model and get posterior probabilities
3) Use an acquistion function, to decide the next sampling point (controls the behavior of search by preferring maxima tracking or reducing overall model uncertainity)
4) Get new observation(s) and proceed to step 2 again.

**(Simply run the python files to see results for different implementations)**

The code has been tested using python version 3.8.0 and Ubuntu 18.04.
Dependencies include but not limited to the python3 version of the following modules:
1) numpy
2) scipy
3) sklearn
4) matplotlib