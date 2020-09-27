# Lamb's problem in two-dimensional elastic medium

This code, written in Python, computes displacements of a semi-infinite elastic
medium subject to a line force.
It only considers responses at a receiver on the free-surface.
The solution is calculated using the convolution integration with the Green's
function.
Two line forces are considered, a Ricker wavelet and Heaviside step loading.
This code is expected to be used as an analytical solution to verify numerical
solutions computed by, for example, finite element methods with direct time
integration scheme.
