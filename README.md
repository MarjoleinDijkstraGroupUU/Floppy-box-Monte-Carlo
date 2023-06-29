# Floppy-box-Monte-Carlo
Floppy-box Monte Carlo simulation script for Hoomd-Blue 3. This script works for rounded truncated tetrahedron and rounded tetrahedron.

## Requires

- hoomd-blue 3
- numpy

## Usage

There are three parameters you need to specify. `-t`: truncation 0-1 (if t=0, means rounded tetrahedron); `-r`: roundness 0-1; `-n`: number of particles in the unit cell.
Of course, you also can change other parameters, like the final pressure and total MC steps.

At least 10 (20 recommended) independent runs are needed to get the densest packing.

## Todo

- box_reduce updater
