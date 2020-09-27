# Histoptimizer

## Overview

Histoptimizer is a Python library and CLI for taking a DataFrame or equivalent, sorting it by a column (optional), and
optimally distributing the ordered rows into a given number of partitions as equally as possible.

The use case that motivated its creation was: Taking a list of the ~3117 counties in the U.S., ordering them by some
attribute (voting averages, population density, median age, etc.), and then distributing them into a number of buckets
of approximately equal population. This makes histograms and other charts that use discrete distributions more accurate.

## Usage

