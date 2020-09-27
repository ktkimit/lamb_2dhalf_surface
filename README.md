<!-- Copyright © 2020 Ki-Tae Kim -->

<!-- This library is free software; you can redistribute it and/or modify -->
<!-- it under the terms of the GNU Lesser General Public License as published -->
<!-- by the Free Software Foundation; either version 3 of the License, or -->
<!-- (at your option) any later version. -->

<!-- This library is distributed in the hope that it will be useful, -->
<!-- but WITHOUT ANY WARRANTY; without even the implied warranty of -->
<!-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the -->
<!-- GNU Lesser General Public License for more details. -->

<!-- You should have received a copy of the GNU Lesser General Public License -->
<!-- along with this library; if not, see <http://www.gnu.org/licenses/>. -->

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
