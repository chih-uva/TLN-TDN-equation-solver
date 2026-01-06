"""
Compute the L=2 source term for h00 using the Mathematica expression provided by the user.

Inputs are NumPy arrays (or scalars) of matching shape for r-dependent quantities that
should already be evaluated on the same grid.
"""

import re
from functools import lru_cache

import numpy as np
from sympy import lambdify
from sympy.parsing.mathematica import parse_mathematica


@lru_cache(maxsize=1)
def _get_lambdified_source():
    # Original Mathematica expression pasted verbatim, then sanitized below.
    raw = r"""
-1/(3 (-1 + E^\[Lambda][r] + 8 E^\[Lambda][r] \[Pi] r^2 p0[r])^6)
   I E^(-\[Lambda][r] - \[Nu][r]/
    2) \[Pi] (r (-1 + E^\[Lambda][r] + 
        8 E^\[Lambda][r] \[Pi] r^2 p0[r]) Derivative[1][h00][
       r] (-2 r (-1 + E^\[Lambda][r] + 
           8 E^\[Lambda][r] \[Pi] r^2 p0[r]) (-8 E^\[Lambda][
             r] r (-1 + E^\[Lambda][r] + 
              8 E^\[Lambda][r] \[Pi] r^2 p0[r]) (-1 + E^(
              2 \[Lambda][r]) + 
              8 E^\[Lambda][r] (-3 + 2 E^\[Lambda][r]) \[Pi] r^2 p0[
                r] + 64 E^(2 \[Lambda][r]) \[Pi]^2 r^4 p0[r]^2 - 
              8 E^\[Lambda][r] \[Pi] r^2 \[Rho]0[r]) Derivative[
             1][\[Rho]0][r]^2 (\[Eta]^\[Prime]\[Prime])[\[Rho]0[r]] + 
           Derivative[
             1][\[Eta]][\[Rho]0[
              r]] (((-1 + E^\[Lambda][r])^2 (1 - 44 E^\[Lambda][r] - 
                    66 E^(2 \[Lambda][r]) + 4 E^(3 \[Lambda][r]) + E^(
                    4 \[Lambda][r])) + 
                 12288 E^(
                  4 \[Lambda][r]) (5 - 6 E^\[Lambda][r] + 
                    5 E^(2 \[Lambda][r])) \[Pi]^4 r^8 p0[r]^4 + 
                 196608 E^(
                  5 \[Lambda][r]) (-1 + 
                    E^\[Lambda][r]) \[Pi]^5 r^10 p0[r]^5 + 
                 262144 E^(6 \[Lambda][r]) \[Pi]^6 r^12 p0[r]^6 + 
                 32 E^(2 \[Lambda][r]) (-37 + 31 E^\[Lambda][r] + 
                    5 E^(2 \[Lambda][r]) + E^(
                    3 \[Lambda][r])) \[Pi] r^2 \[Rho]0[r] - 
                 256 E^(3 \[Lambda][r]) (11 + 
                    E^\[Lambda][r]) \[Pi]^2 r^4 \[Rho]0[r]^2 + 
                 512 E^(3 \[Lambda][r]) \[Pi]^3 r^6 p0[
                   r]^3 (-20 + 20 E^\[Lambda][r] - 
                    16 E^(2 \[Lambda][r]) + 20 E^(3 \[Lambda][r]) + 
                    32 E^(2 \[Lambda][r]) \[Pi] r^2 \[Rho]0[r]) + 
                 64 E^(2 \[Lambda][r]) \[Pi]^2 r^4 p0[
                   r]^2 (15 + 28 E^\[Lambda][r] - 
                    66 E^(2 \[Lambda][r]) + 15 E^(4 \[Lambda][r]) + 
                    32 E^(2 \[Lambda][r]) (1 + 
                    3 E^\[Lambda][r]) \[Pi] r^2 \[Rho]0[r]) + 
                 8 E^\[Lambda][r] \[Pi] r^2 p0[
                   r] ((-1 + E^\[Lambda][r]) (6 + 
                    428 E^\[Lambda][r] - 132 E^(2 \[Lambda][r]) + 
                    12 E^(3 \[Lambda][r]) + 6 E^(4 \[Lambda][r])) + 
                    32 E^(2 \[Lambda][r]) (-37 + 6 E^\[Lambda][r] + 
                    3 E^(2 \[Lambda][r])) \[Pi] r^2 \[Rho]0[r] - 
                    256 E^(3 \[Lambda][r]) \[Pi]^2 r^4 \[Rho]0[
                    r]^2)) Derivative[1][\[Rho]0][r] + 
              128 E^(2 \[Lambda][r]) \[Pi] r^3 (-1 + E^\[Lambda][r] + 
                 8 E^\[Lambda][r] \[Pi] r^2 p0[r]) Derivative[
                1][\[Rho]0][r]^2 - 
              8 E^\[Lambda][
                r] r (-1 + E^\[Lambda][r] + 
                 8 E^\[Lambda][r] \[Pi] r^2 p0[r]) (-1 + E^(
                 2 \[Lambda][r]) + 
                 8 E^\[Lambda][
                   r] (-3 + 2 E^\[Lambda][r]) \[Pi] r^2 p0[r] + 
                 64 E^(2 \[Lambda][r]) \[Pi]^2 r^4 p0[r]^2 - 
                 8 E^\[Lambda][r] \[Pi] r^2 \[Rho]0[
                   r]) (\[Rho]0^\[Prime]\[Prime])[
                r])) + \[Eta][\[Rho]0[
           r]] (25165824 E^(
            7 \[Lambda][r]) (-3 + 2 E^\[Lambda][r]) \[Pi]^7 r^14 p0[
             r]^7 + 50331648 E^(
            8 \[Lambda][r]) \[Pi]^8 r^16 p0[r]^8 + 
           2560 E^(3 \[Lambda][r]) (-23 + 22 E^\[Lambda][r] + E^(
              2 \[Lambda][r])) \[Pi]^2 r^4 \[Rho]0[r]^2 - 
           98304 E^(4 \[Lambda][r]) \[Pi]^3 r^6 \[Rho]0[r]^3 + 
           262144 E^(
            6 \[Lambda][r]) \[Pi]^6 r^12 p0[
             r]^6 (156 - 228 E^\[Lambda][r] + 84 E^(2 \[Lambda][r]) - 
              32 E^\[Lambda][r] \[Pi] r^2 \[Rho]0[r]) + 
           61440 E^(
            4 \[Lambda][r]) (-1 + E^\[Lambda][r]) \[Pi]^4 r^8 p0[
             r]^4 (-30 + 46 E^\[Lambda][r] - 46 E^(2 \[Lambda][r]) + 
              14 E^(3 \[Lambda][r]) - 
              32 E^\[Lambda][
                r] (-1 + E^\[Lambda][r]) \[Pi] r^2 \[Rho]0[r]) + 
           196608 E^(
            5 \[Lambda][r]) \[Pi]^5 r^10 p0[
             r]^5 (-58 + 124 E^\[Lambda][r] - 
              102 E^(2 \[Lambda][r]) + 28 E^(3 \[Lambda][r]) - 
              32 E^\[Lambda][
                r] (-1 + E^\[Lambda][r]) \[Pi] r^2 \[Rho]0[r]) + 
           2048 E^(3 \[Lambda][r]) \[Pi]^3 r^6 p0[
             r]^3 (3 (-1 + E^\[Lambda][r]) (29 - 81 E^\[Lambda][r] + 
                 49 E^(2 \[Lambda][r]) - 51 E^(3 \[Lambda][r]) + 
                 14 E^(4 \[Lambda][r])) - 
              80 E^\[Lambda][
                r] (-2 + 2 E^\[Lambda][r] - 6 E^(2 \[Lambda][r]) + 
                 2 E^(3 \[Lambda][r])) \[Pi] r^2 \[Rho]0[r] - 
              64 E^(2 \[Lambda][r]) \[Pi] r^3 Derivative[1][\[Rho]0][
                r]) - 32 E^\[Lambda][
             r] (-1 + E^\[Lambda][r]) \[Pi] r^2 \[Rho]0[
             r] ((-1 + E^\[Lambda][r]) (1 + 264 E^\[Lambda][r] + 
                 42 E^(2 \[Lambda][r]) - 4 E^(3 \[Lambda][r]) + E^(
                 4 \[Lambda][r])) - 
              16 E^(2 \[Lambda][r]) (17 + 
                 E^\[Lambda][r]) \[Pi] r^3 Derivative[1][\[Rho]0][
                r]) + (-1 + 
              E^\[Lambda][
               r])^2 (3 (-1 + E^\[Lambda][r]) (-1 + 
                 25 E^\[Lambda][r] + 58 E^(2 \[Lambda][r]) - 
                 18 E^(3 \[Lambda][r]) - E^(4 \[Lambda][r]) + E^(
                 5 \[Lambda][r])) - 
              64 E^(2 \[Lambda][r]) (31 + 
                 5 E^\[Lambda][r]) \[Pi] r^3 Derivative[1][\[Rho]0][
                r] - 
              128 E^(2 \[Lambda][
                 r]) \[Pi] r^4 (\[Rho]0^\[Prime]\[Prime])[r]) + 
           64 E^(2 \[Lambda][r]) \[Pi]^2 r^4 p0[
             r]^2 (156 - 3012 E^\[Lambda][r] + 
              3516 E^(2 \[Lambda][r]) - 768 E^(3 \[Lambda][r]) + 
              420 E^(4 \[Lambda][r]) - 396 E^(5 \[Lambda][r]) + 
              84 E^(6 \[Lambda][r]) + 
              2560 E^(3 \[Lambda][r]) \[Pi]^2 r^4 \[Rho]0[r]^2 - 
              64 E^(2 \[Lambda][r]) (-31 + 
                 13 E^\[Lambda][r]) \[Pi] r^3 Derivative[1][\[Rho]0][
                r] - 32 E^\[Lambda][r] \[Pi] r^2 \[Rho]0[
                r] (15 + 144 E^\[Lambda][r] + 46 E^(2 \[Lambda][r]) - 
                 60 E^(3 \[Lambda][r]) + 15 E^(4 \[Lambda][r]) - 
                 16 E^(2 \[Lambda][r]) \[Pi] r^3 Derivative[
                   1][\[Rho]0][r]) - 
              128 E^(2 \[Lambda][
                 r]) \[Pi] r^4 (\[Rho]0^\[Prime]\[Prime])[r]) + 
           16 E^\[Lambda][r] \[Pi] r^2 p0[
             r] (512 E^(
               3 \[Lambda][r]) (-53 + 
                 5 E^\[Lambda][r]) \[Pi]^2 r^4 \[Rho]0[r]^2 - 
              32 E^\[Lambda][r] \[Pi] r^2 \[Rho]0[
                r] ((-1 + E^\[Lambda][r]) (3 - 444 E^\[Lambda][r] + 
                    34 E^(2 \[Lambda][r]) - 12 E^(3 \[Lambda][r]) + 
                    3 E^(4 \[Lambda][r])) - 
                 16 E^(2 \[Lambda][r]) (8 + 
                    E^\[Lambda][r]) \[Pi] r^3 Derivative[1][\[Rho]0][
                   r]) + (-1 + 
                 E^\[Lambda][
                  r]) (3 (-1 + E^\[Lambda][r]) (-6 - 
                    480 E^\[Lambda][r] + 8 E^(2 \[Lambda][r]) - 
                    28 E^(3 \[Lambda][r]) - 10 E^(4 \[Lambda][r]) + 
                    4 E^(5 \[Lambda][r])) - 
                 64 E^(2 \[Lambda][r]) (2 + 
                    7 E^\[Lambda][r]) \[Pi] r^3 Derivative[
                   1][\[Rho]0][r] - 
                 128 E^(2 \[Lambda][
                    r]) \[Pi] r^4 (\[Rho]0^\[Prime]\[Prime])[r])))) + 
     h00[r] (-2 r (-1 + E^\[Lambda][r] + 
           8 E^\[Lambda][r] \[Pi] r^2 p0[r]) (-8 E^\[Lambda][
             r] r (-1 + E^\[Lambda][r] + 
              8 E^\[Lambda][r] \[Pi] r^2 p0[
                r]) ((-1 + E^\[Lambda][r])^2 (1 + 6 E^\[Lambda][r] + 
                 E^(2 \[Lambda][r])) + 
              1024 E^(3 \[Lambda][r]) (-3 + 
                 2 E^\[Lambda][r]) \[Pi]^3 r^6 p0[r]^3 + 
              4096 E^(4 \[Lambda][r]) \[Pi]^4 r^8 p0[r]^4 - 
              16 E^\[Lambda][
                r] (-1 + E^(2 \[Lambda][r])) \[Pi] r^2 \[Rho]0[r] + 
              64 E^(2 \[Lambda][r]) \[Pi]^2 r^4 \[Rho]0[r]^2 + 
              64 E^(2 \[Lambda][r]) \[Pi]^2 r^4 p0[
                r]^2 (7 - 8 E^\[Lambda][r] + 6 E^(2 \[Lambda][r]) - 
                 16 E^\[Lambda][r] \[Pi] r^2 \[Rho]0[r]) + 
              16 E^\[Lambda][r] \[Pi] r^2 p0[
                r] ((-1 + E^\[Lambda][r]) (-3 + 3 E^\[Lambda][r] + 
                    2 E^(2 \[Lambda][r])) - 
                 8 E^\[Lambda][
                   r] (-3 + 2 E^\[Lambda][r]) \[Pi] r^2 \[Rho]0[
                   r])) Derivative[1][\[Rho]0][
             r]^2 (\[Eta]^\[Prime]\[Prime])[\[Rho]0[r]] + 
           Derivative[
             1][\[Eta]][\[Rho]0[
              r]] ((-1 + 42 E^\[Lambda][r] + 170 E^(2 \[Lambda][r]) - 
                 702 E^(3 \[Lambda][r]) + 672 E^(4 \[Lambda][r]) - 
                 114 E^(5 \[Lambda][r]) - 74 E^(6 \[Lambda][r]) + 
                 6 E^(7 \[Lambda][r]) + E^(8 \[Lambda][r]) + 
                 2097152 E^(
                  7 \[Lambda][r]) (-9 + 
                    8 E^\[Lambda][r]) \[Pi]^7 r^14 p0[r]^7 + 
                 16777216 E^(8 \[Lambda][r]) \[Pi]^8 r^16 p0[r]^8 - 
                 8 E^\[Lambda][r] \[Pi] r^2 \[Rho]0[r] + 
                 1552 E^(2 \[Lambda][r]) \[Pi] r^2 \[Rho]0[r] - 
                 792 E^(3 \[Lambda][r]) \[Pi] r^2 \[Rho]0[r] - 
                 2720 E^(4 \[Lambda][r]) \[Pi] r^2 \[Rho]0[r] + 
                 1672 E^(5 \[Lambda][r]) \[Pi] r^2 \[Rho]0[r] + 
                 272 E^(6 \[Lambda][r]) \[Pi] r^2 \[Rho]0[r] + 
                 24 E^(7 \[Lambda][r]) \[Pi] r^2 \[Rho]0[r] + 
                 12288 E^(3 \[Lambda][r]) \[Pi]^2 r^4 \[Rho]0[r]^2 - 
                 7680 E^(4 \[Lambda][r]) \[Pi]^2 r^4 \[Rho]0[r]^2 - 
                 4096 E^(5 \[Lambda][r]) \[Pi]^2 r^4 \[Rho]0[r]^2 - 
                 512 E^(6 \[Lambda][r]) \[Pi]^2 r^4 \[Rho]0[r]^2 + 
                 22528 E^(4 \[Lambda][r]) \[Pi]^3 r^6 \[Rho]0[r]^3 + 
                 2048 E^(5 \[Lambda][r]) \[Pi]^3 r^6 \[Rho]0[r]^3 + 
                 1048576 E^(
                  6 \[Lambda][r]) \[Pi]^6 r^12 p0[
                   r]^6 (8 - 11 E^\[Lambda][r] + 
                    7 E^(2 \[Lambda][r]) - 
                    2 E^\[Lambda][r] \[Pi] r^2 \[Rho]0[r]) + 
                 32768 E^(
                  5 \[Lambda][r]) \[Pi]^5 r^10 p0[
                   r]^5 ((-1 + E^\[Lambda][r]) (59 - 
                    23 E^\[Lambda][r] + 56 E^(2 \[Lambda][r])) + 
                    8 E^\[Lambda][
                    r] (6 - 2 E^\[Lambda][r]) \[Pi] r^2 \[Rho]0[r]) + 
                 4096 E^(4 \[Lambda][r]) \[Pi]^4 r^8 p0[
                   r]^4 (60 - 34 E^\[Lambda][r] - 
                    26 E^(2 \[Lambda][r]) - 50 E^(3 \[Lambda][r]) + 
                    70 E^(4 \[Lambda][r]) + 
                    40 E^\[Lambda][
                    r] (-1 + E^\[Lambda][r]) (3 + 
                    E^\[Lambda][r]) \[Pi] r^2 \[Rho]0[r]) + 
                 512 E^(3 \[Lambda][r]) \[Pi]^3 r^6 p0[
                   r]^3 (-31 - 496 E^\[Lambda][r] + 
                    810 E^(2 \[Lambda][r]) - 324 E^(3 \[Lambda][r]) + 
                    25 E^(4 \[Lambda][r]) + 56 E^(5 \[Lambda][r]) + 
                    8 E^\[Lambda][
                    r] (20 - 184 E^\[Lambda][r] + 
                    28 E^(2 \[Lambda][r]) + 
                    20 E^(3 \[Lambda][r])) \[Pi] r^2 \[Rho]0[r] - 
                    512 E^(3 \[Lambda][r]) \[Pi]^2 r^4 \[Rho]0[
                    r]^2) + 64 E^(
                  2 \[Lambda][r]) \[Pi]^2 r^4 p0[
                   r]^2 ((-1 + E^\[Lambda][r]) (-4 - 
                    1164 E^\[Lambda][r] + 1016 E^(2 \[Lambda][r]) - 
                    392 E^(3 \[Lambda][r]) + 84 E^(4 \[Lambda][r]) + 
                    28 E^(5 \[Lambda][r])) + 
                    8 E^\[Lambda][
                    r] (-15 + 264 E^\[Lambda][r] - 
                    174 E^(2 \[Lambda][r]) + 84 E^(3 \[Lambda][r]) + 
                    25 E^(4 \[Lambda][r])) \[Pi] r^2 \[Rho]0[r] - 
                    768 E^(3 \[Lambda][r]) (3 + 
                    2 E^\[Lambda][r]) \[Pi]^2 r^4 \[Rho]0[r]^2) + 
                 8 E^\[Lambda][r] \[Pi] r^2 p0[
                   r] ((-1 + E^\[Lambda][r])^2 (3 + 
                    584 E^\[Lambda][r] + 30 E^(2 \[Lambda][r]) - 
                    216 E^(3 \[Lambda][r]) + 47 E^(4 \[Lambda][r]) + 
                    8 E^(5 \[Lambda][r])) + 
                    8 E^\[Lambda][
                    r] (-1 + E^\[Lambda][r]) (-6 - 
                    1020 E^\[Lambda][r] + 312 E^(2 \[Lambda][r]) + 
                    108 E^(3 \[Lambda][r]) + 
                    14 E^(4 \[Lambda][r])) \[Pi] r^2 \[Rho]0[r] - 
                    256 E^(3 \[Lambda][r]) (-71 + 25 E^\[Lambda][r] + 
                    6 E^(2 \[Lambda][r])) \[Pi]^2 r^4 \[Rho]0[r]^2 + 
                    2048 E^(4 \[Lambda][r]) \[Pi]^3 r^6 \[Rho]0[
                    r]^3)) Derivative[1][\[Rho]0][r] + 
              128 E^(2 \[Lambda][r]) \[Pi] r^3 (-1 + E^\[Lambda][r] + 
                 8 E^\[Lambda][r] \[Pi] r^2 p0[r]) (-1 + E^(
                 2 \[Lambda][r]) + 
                 8 E^\[Lambda][
                   r] (-3 + 2 E^\[Lambda][r]) \[Pi] r^2 p0[r] + 
                 64 E^(2 \[Lambda][r]) \[Pi]^2 r^4 p0[r]^2 - 
                 8 E^\[Lambda][r] \[Pi] r^2 \[Rho]0[r]) Derivative[
                1][\[Rho]0][r]^2 - 
              8 E^\[Lambda][
                r] r (-1 + E^\[Lambda][r] + 
                 8 E^\[Lambda][r] \[Pi] r^2 p0[
                   r]) ((-1 + E^\[Lambda][r])^2 (1 + 
                    6 E^\[Lambda][r] + E^(2 \[Lambda][r])) + 
                 1024 E^(3 \[Lambda][r]) (-3 + 
                    2 E^\[Lambda][r]) \[Pi]^3 r^6 p0[r]^3 + 
                 4096 E^(4 \[Lambda][r]) \[Pi]^4 r^8 p0[r]^4 - 
                 16 E^\[Lambda][
                   r] (-1 + E^(2 \[Lambda][r])) \[Pi] r^2 \[Rho]0[
                   r] + 64 E^(
                  2 \[Lambda][r]) \[Pi]^2 r^4 \[Rho]0[r]^2 + 
                 64 E^(2 \[Lambda][r]) \[Pi]^2 r^4 p0[
                   r]^2 (7 - 8 E^\[Lambda][r] + 
                    6 E^(2 \[Lambda][r]) - 
                    16 E^\[Lambda][r] \[Pi] r^2 \[Rho]0[r]) + 
                 16 E^\[Lambda][r] \[Pi] r^2 p0[
                   r] ((-1 + E^\[Lambda][r]) (-3 + 3 E^\[Lambda][r] + 
                    2 E^(2 \[Lambda][r])) - 
                    8 E^\[Lambda][
                    r] (-3 + 2 E^\[Lambda][r]) \[Pi] r^2 \[Rho]0[
                    r])) (\[Rho]0^\[Prime]\[Prime])[
                r])) + \[Eta][\[Rho]0[r]] (-3 + 72 E^\[Lambda][r] + 
           351 E^(2 \[Lambda][r]) - 1944 E^(3 \[Lambda][r]) + 
           2898 E^(4 \[Lambda][r]) - 1512 E^(5 \[Lambda][r]) - 
           114 E^(6 \[Lambda][r]) + 312 E^(7 \[Lambda][r]) - 
           63 E^(8 \[Lambda][r]) + 3 E^(10 \[Lambda][r]) + 
           2013265920 E^(
            9 \[Lambda][r]) (-3 + 2 E^\[Lambda][r]) \[Pi]^9 r^18 p0[
             r]^9 + 3221225472 E^(
            10 \[Lambda][r]) \[Pi]^10 r^20 p0[r]^10 + 
           8 E^\[Lambda][r] \[Pi] r^2 \[Rho]0[r] + 
           9184 E^(2 \[Lambda][r]) \[Pi] r^2 \[Rho]0[r] - 
           9632 E^(3 \[Lambda][r]) \[Pi] r^2 \[Rho]0[r] - 
           25376 E^(4 \[Lambda][r]) \[Pi] r^2 \[Rho]0[r] + 
           41968 E^(5 \[Lambda][r]) \[Pi] r^2 \[Rho]0[r] - 
           15200 E^(6 \[Lambda][r]) \[Pi] r^2 \[Rho]0[r] - 
           1056 E^(7 \[Lambda][r]) \[Pi] r^2 \[Rho]0[r] + 
           160 E^(8 \[Lambda][r]) \[Pi] r^2 \[Rho]0[r] - 
           56 E^(9 \[Lambda][r]) \[Pi] r^2 \[Rho]0[r] + 
           256 E^(2 \[Lambda][r]) \[Pi]^2 r^4 \[Rho]0[r]^2 + 
           125952 E^(3 \[Lambda][r]) \[Pi]^2 r^4 \[Rho]0[r]^2 - 
           164096 E^(4 \[Lambda][r]) \[Pi]^2 r^4 \[Rho]0[r]^2 - 
           49152 E^(5 \[Lambda][r]) \[Pi]^2 r^4 \[Rho]0[r]^2 + 
           85760 E^(6 \[Lambda][r]) \[Pi]^2 r^4 \[Rho]0[r]^2 + 
           1024 E^(7 \[Lambda][r]) \[Pi]^2 r^4 \[Rho]0[r]^2 + 
           256 E^(8 \[Lambda][r]) \[Pi]^2 r^4 \[Rho]0[r]^2 + 
           569344 E^(4 \[Lambda][r]) \[Pi]^3 r^6 \[Rho]0[r]^3 - 
           450560 E^(5 \[Lambda][r]) \[Pi]^3 r^6 \[Rho]0[r]^3 - 
           118784 E^(6 \[Lambda][r]) \[Pi]^3 r^6 \[Rho]0[r]^3 + 
           786432 E^(5 \[Lambda][r]) \[Pi]^4 r^8 \[Rho]0[r]^4 + 
           16777216 E^(
            8 \[Lambda][r]) \[Pi]^8 r^16 p0[
             r]^8 (261 - 360 E^\[Lambda][r] + 
              135 E^(2 \[Lambda][r]) - 
              56 E^\[Lambda][r] \[Pi] r^2 \[Rho]0[r]) + 
           2097152 E^(
            7 \[Lambda][r]) \[Pi]^7 r^14 p0[
             r]^7 (-780 + 1596 E^\[Lambda][r] - 
              1260 E^(2 \[Lambda][r]) + 360 E^(3 \[Lambda][r]) - 
              64 E^\[Lambda][
                r] (-9 + 7 E^\[Lambda][r]) \[Pi] r^2 \[Rho]0[r]) + 
           524288 E^(
            6 \[Lambda][r]) \[Pi]^6 r^12 p0[
             r]^6 (669 - 1740 E^\[Lambda][r] + 
              2016 E^(2 \[Lambda][r]) - 1260 E^(3 \[Lambda][r]) + 
              315 E^(4 \[Lambda][r]) - 
              4 E^\[Lambda][
                r] (284 - 452 E^\[Lambda][r] + 
                 196 E^(2 \[Lambda][r])) \[Pi] r^2 \[Rho]0[r] + 
              128 E^(2 \[Lambda][r]) \[Pi]^2 r^4 \[Rho]0[r]^2) + 
           1984 E^(2 \[Lambda][r]) \[Pi] r^3 Derivative[1][\[Rho]0][
             r] - 2624 E^(
            3 \[Lambda][r]) \[Pi] r^3 Derivative[1][\[Rho]0][r] - 
           3712 E^(4 \[Lambda][r]) \[Pi] r^3 Derivative[1][\[Rho]0][
             r] + 7040 E^(
            5 \[Lambda][r]) \[Pi] r^3 Derivative[1][\[Rho]0][r] - 
           2368 E^(6 \[Lambda][r]) \[Pi] r^3 Derivative[1][\[Rho]0][
             r] - 320 E^(
            7 \[Lambda][r]) \[Pi] r^3 Derivative[1][\[Rho]0][r] + 
           24576 E^(
            3 \[Lambda][r]) \[Pi]^2 r^5 \[Rho]0[r] Derivative[
             1][\[Rho]0][r] - 
           37376 E^(
            4 \[Lambda][r]) \[Pi]^2 r^5 \[Rho]0[r] Derivative[
             1][\[Rho]0][r] + 
           1536 E^(5 \[Lambda][r]) \[Pi]^2 r^5 \[Rho]0[r] Derivative[
             1][\[Rho]0][r] + 
           10752 E^(
            6 \[Lambda][r]) \[Pi]^2 r^5 \[Rho]0[r] Derivative[
             1][\[Rho]0][r] + 
           512 E^(7 \[Lambda][r]) \[Pi]^2 r^5 \[Rho]0[r] Derivative[
             1][\[Rho]0][r] + 
           69632 E^(
            4 \[Lambda][r]) \[Pi]^3 r^7 \[Rho]0[r]^2 Derivative[
             1][\[Rho]0][r] - 
           65536 E^(
            5 \[Lambda][r]) \[Pi]^3 r^7 \[Rho]0[r]^2 Derivative[
             1][\[Rho]0][r] - 
           4096 E^(6 \[Lambda][r]) \[Pi]^3 r^7 \[Rho]0[
             r]^2 Derivative[1][\[Rho]0][r] + 
           32768 E^(
            5 \[Lambda][r]) \[Pi]^5 r^10 p0[
             r]^5 (3 (-1 + E^\[Lambda][r]) (450 - 
                 1046 E^\[Lambda][r] + 966 E^(2 \[Lambda][r]) - 
                 798 E^(3 \[Lambda][r]) + 252 E^(4 \[Lambda][r])) - 
              16 E^\[Lambda][
                r] (-292 + 608 E^\[Lambda][r] - 
                 600 E^(2 \[Lambda][r]) + 
                 196 E^(3 \[Lambda][r])) \[Pi] r^2 \[Rho]0[r] + 
              1536 E^(2 \[Lambda][r]) (-1 + 
                 E^\[Lambda][r]) \[Pi]^2 r^4 \[Rho]0[r]^2 - 
              256 E^(2 \[Lambda][r]) \[Pi] r^3 Derivative[1][\[Rho]0][
                r]) + 128 E^(
            2 \[Lambda][r]) \[Pi] r^4 (\[Rho]0^\[Prime]\[Prime])[r] - 
           256 E^(3 \[Lambda][
              r]) \[Pi] r^4 (\[Rho]0^\[Prime]\[Prime])[r] + 
           256 E^(5 \[Lambda][
              r]) \[Pi] r^4 (\[Rho]0^\[Prime]\[Prime])[r] - 
           128 E^(6 \[Lambda][
              r]) \[Pi] r^4 (\[Rho]0^\[Prime]\[Prime])[r] + 
           1024 E^(3 \[Lambda][r]) \[Pi]^2 r^6 \[Rho]0[
             r] (\[Rho]0^\[Prime]\[Prime])[r] - 
           2048 E^(4 \[Lambda][r]) \[Pi]^2 r^6 \[Rho]0[
             r] (\[Rho]0^\[Prime]\[Prime])[r] + 
           1024 E^(5 \[Lambda][r]) \[Pi]^2 r^6 \[Rho]0[
             r] (\[Rho]0^\[Prime]\[Prime])[r] + 
           8192 E^(4 \[Lambda][r]) \[Pi]^4 r^8 p0[
             r]^4 (375 - 3264 E^\[Lambda][r] + 
              4365 E^(2 \[Lambda][r]) - 2316 E^(3 \[Lambda][r]) + 
              1785 E^(4 \[Lambda][r]) - 1260 E^(5 \[Lambda][r]) + 
              315 E^(6 \[Lambda][r]) + 
              320 E^(2 \[Lambda][r]) (6 - 8 E^\[Lambda][r] + 
                 6 E^(2 \[Lambda][r])) \[Pi]^2 r^4 \[Rho]0[r]^2 - 
              32 E^(2 \[Lambda][r]) (-43 + 
                 21 E^\[Lambda][r]) \[Pi] r^3 Derivative[1][\[Rho]0][
                r] - 4 E^\[Lambda][r] \[Pi] r^2 \[Rho]0[
                r] (690 - 764 E^\[Lambda][r] + 
                 2156 E^(2 \[Lambda][r]) - 1740 E^(3 \[Lambda][r]) + 
                 490 E^(4 \[Lambda][r]) - 
                 64 E^(2 \[Lambda][r]) \[Pi] r^3 Derivative[
                   1][\[Rho]0][r]) - 
              64 E^(2 \[Lambda][
                 r]) \[Pi] r^4 (\[Rho]0^\[Prime]\[Prime])[r]) + 
           512 E^(3 \[Lambda][r]) \[Pi]^3 r^6 p0[
             r]^3 (-156 + 5628 E^\[Lambda][r] - 
              8244 E^(2 \[Lambda][r]) + 3384 E^(3 \[Lambda][r]) - 
              468 E^(4 \[Lambda][r]) + 756 E^(5 \[Lambda][r]) - 
              1260 E^(6 \[Lambda][r]) + 360 E^(7 \[Lambda][r]) + 
              128 E^(2 \[Lambda][r]) (-40 - 444 E^\[Lambda][r] - 
                 40 E^(2 \[Lambda][r]) + 
                 40 E^(3 \[Lambda][r])) \[Pi]^2 r^4 \[Rho]0[r]^2 - 
              64 E^(2 \[Lambda][r]) (85 - 95 E^\[Lambda][r] + 
                 44 E^(2 \[Lambda][r])) \[Pi] r^3 Derivative[
                1][\[Rho]0][r] - 
              32 E^\[Lambda][r] \[Pi] r^2 \[Rho]0[
                r] (-118 + 802 E^\[Lambda][r] - 
                 1268 E^(2 \[Lambda][r]) + 536 E^(3 \[Lambda][r]) - 
                 370 E^(4 \[Lambda][r]) + 98 E^(5 \[Lambda][r]) - 
                 16 E^(2 \[Lambda][r]) (17 + 
                    4 E^\[Lambda][r]) \[Pi] r^3 Derivative[
                   1][\[Rho]0][r]) + 
              640 E^(2 \[Lambda][
                 r]) \[Pi] r^4 (\[Rho]0^\[Prime]\[Prime])[r] - 
              512 E^(3 \[Lambda][
                 r]) \[Pi] r^4 (\[Rho]0^\[Prime]\[Prime])[r]) + 
           8 E^\[Lambda][r] \[Pi] r^2 p0[
             r] (-8192 E^(
               4 \[Lambda][r]) (-89 + 
                 29 E^\[Lambda][r]) \[Pi]^3 r^6 \[Rho]0[r]^3 + 
              128 E^(2 \[Lambda][r]) \[Pi]^2 r^4 \[Rho]0[
                r]^2 ((-1 + E^\[Lambda][r]) (12 - 
                    3580 E^\[Lambda][r] + 868 E^(2 \[Lambda][r]) + 
                    32 E^(3 \[Lambda][r]) + 12 E^(4 \[Lambda][r])) - 
                 64 E^(2 \[Lambda][r]) (8 + 
                    E^\[Lambda][r]) \[Pi] r^3 Derivative[1][\[Rho]0][
                   r]) - 16 E^\[Lambda][
                r] (-1 + E^\[Lambda][r]) \[Pi] r^2 \[Rho]0[
                r] ((-1 + E^\[Lambda][r]) (-12 - 
                    4752 E^\[Lambda][r] - 316 E^(2 \[Lambda][r]) + 
                    196 E^(3 \[Lambda][r]) - 40 E^(4 \[Lambda][r]) + 
                    28 E^(5 \[Lambda][r])) - 
                 32 E^(2 \[Lambda][r]) (-31 + 63 E^\[Lambda][r] + 
                    4 E^(2 \[Lambda][r])) \[Pi] r^3 Derivative[
                   1][\[Rho]0][r] - 
                 128 E^(2 \[Lambda][
                    r]) \[Pi] r^4 (\[Rho]0^\[Prime]\[Prime])[
                   r]) + (-1 + 
                 E^\[Lambda][
                  r])^2 (3 (-1 + E^\[Lambda][r]) (-9 - 
                    1089 E^\[Lambda][r] - 1428 E^(2 \[Lambda][r]) + 
                    106 E^(3 \[Lambda][r]) - 101 E^(4 \[Lambda][r]) + 
                    15 E^(5 \[Lambda][r]) + 10 E^(6 \[Lambda][r])) - 
                 64 E^(2 \[Lambda][r]) (-89 + 113 E^\[Lambda][r] + 
                    24 E^(2 \[Lambda][r])) \[Pi] r^3 Derivative[
                   1][\[Rho]0][r] - 
                 128 E^(2 \[Lambda][r]) (-1 + 
                    4 E^\[Lambda][
                    r]) \[Pi] r^4 (\[Rho]0^\[Prime]\[Prime])[r])) + 
           64 E^(2 \[Lambda][r]) \[Pi]^2 r^4 p0[
             r]^2 (-118784 E^(
               4 \[Lambda][r]) \[Pi]^3 r^6 \[Rho]0[r]^3 + 
              256 E^(2 \[Lambda][r]) \[Pi]^2 r^4 \[Rho]0[
                r]^2 (15 + 540 E^\[Lambda][r] - 
                 154 E^(2 \[Lambda][r]) + 15 E^(4 \[Lambda][r]) - 
                 16 E^(2 \[Lambda][r]) \[Pi] r^3 Derivative[
                   1][\[Rho]0][r]) - 
              8 E^\[Lambda][r] \[Pi] r^2 \[Rho]0[
                r] ((-1 + E^\[Lambda][r]) (-172 + 
                    13224 E^\[Lambda][r] - 7428 E^(2 \[Lambda][r]) + 
                    860 E^(3 \[Lambda][r]) - 536 E^(4 \[Lambda][r]) + 
                    196 E^(5 \[Lambda][r])) - 
                 64 E^(2 \[Lambda][r]) (-97 + 55 E^\[Lambda][r] + 
                    6 E^(2 \[Lambda][r])) \[Pi] r^3 Derivative[
                   1][\[Rho]0][r] - 
                 128 E^(2 \[Lambda][
                    r]) \[Pi] r^4 (\[Rho]0^\[Prime]\[Prime])[
                   r]) + (-1 + 
                 E^\[Lambda][r]) (-128 E^(
                  2 \[Lambda][r]) (-37 + 11 E^\[Lambda][r] + 
                    23 E^(2 \[Lambda][r])) \[Pi] r^3 Derivative[
                   1][\[Rho]0][r] + 
                 3 (-1 + E^\[Lambda][r]) (-15 + 3610 E^\[Lambda][r] - 
                    2285 E^(2 \[Lambda][r]) + 
                    100 E^(3 \[Lambda][r]) - 273 E^(4 \[Lambda][r]) - 
                    30 E^(5 \[Lambda][r]) + 45 E^(6 \[Lambda][r]) - 
                    256 E^(2 \[Lambda][
                    r]) \[Pi] r^4 (\[Rho]0^\[Prime]\[Prime])[r]))))))
"""
    # Flatten whitespace to make regex replacements robust
    expr_str = re.sub(r"\s+", " ", raw)

    # Normalize prime markers early.
    expr_str = expr_str.replace(r"\[Prime]", "Prime")

    # First, drop explicit Mathematica symbol wrappers.
    simple_tokens = {
        r"\[Lambda]": "lamb",
        r"\[Nu]": "nu",
        r"\[Pi]": "Pi",
        r"\[Rho]0": "rho0",
        r"\[Eta]": "eta",
    }
    for old, new in simple_tokens.items():
        expr_str = expr_str.replace(old, new)

    replacements = [
        (r"Derivative\s*\[\s*1\s*\]\s*\[\s*h00\s*\]\s*\[\s*r\s*\]", "dh00"),
        (r"h00\s*\[\s*r\s*\]", "h00"),
        (r"p0\s*\[\s*r\s*\]", "p0"),
        (r"rho0\s*\[\s*r\s*\]", "rho0"),
        (r"Derivative\s*\[\s*1\s*\]\s*\[\s*rho0\s*\]\s*\[\s*r\s*\]", "drho0"),
        (r"\(\s*rho0\^\s*Prime\s*Prime\s*\)\s*\[\s*r\s*\]", "d2rho0"),
        (r"\(\s*eta\^\s*Prime\s*Prime\s*\)\s*\[\s*rho0\s*\[\s*r\s*\]\s*\]", "d2eta"),
        (r"Derivative\s*\[\s*1\s*\]\s*\[\s*eta\s*\]\s*\[\s*rho0\s*\[\s*r\s*\]\s*\]", "deta"),
        (r"eta\s*\[\s*rho0\s*\[\s*r\s*\]\s*\]", "eta"),
        (r"lamb\s*\[\s*r\s*\]", "lamb"),
        (r"nu\s*\[\s*r\s*\]", "nu"),
    ]

    for pattern, repl in replacements:
        expr_str = re.sub(pattern, repl, expr_str)

    # Clean remaining eta derivative tokens that retain explicit prime markers.
    expr_str = expr_str.replace("(eta^[Prime][Prime])[rho0]", "d2eta")
    expr_str = expr_str.replace("eta^PrimePrime", "d2eta")
    expr_str = expr_str.replace("Derivative[ 1][eta][rho0]", "deta")
    # Drop lingering function-call syntax on eta/deta/d2eta.
    expr_str = re.sub(r"d2eta\s*\(\s*rho0\s*\)", "d2eta", expr_str)
    expr_str = re.sub(r"deta\s*\(\s*rho0\s*\)", "deta", expr_str)
    expr_str = re.sub(r"eta\s*\(\s*rho0\s*\)", "eta", expr_str)

    # Mathematica uses implicit multiplication; the parser understands it. Keep ^ syntax.
    expr = parse_mathematica(expr_str)

    args = (
        "r",
        "p0",
        "rho0",
        "drho0",
        "d2rho0",
        "h00",
        "dh00",
        "lamb",
        "nu",
        "eta",
        "deta",
        "d2eta",
    )
    symbols = {name: parse_mathematica(name) for name in args}

    fn = lambdify(
        tuple(symbols[name] for name in args),
        expr,
        modules=["numpy"],
    )
    return fn


class _CallableArray:
    """
    Wrap an array/scalar so it behaves both as a callable f(rho) and a numeric value.
    This allows eta/deta/d2eta to be passed as arrays without hitting callable errors.
    """

    def __init__(self, value):
        self.value = np.asarray(value)

    def __call__(self, rho):
        return np.broadcast_to(self.value, np.shape(rho))

    def __array__(self, dtype=None):
        return np.asarray(self.value, dtype=dtype)

    # Basic arithmetic to play nicely with NumPy ufuncs
    def _binary_op(self, other, op):
        return op(np.asarray(self.value), other)

    __mul__ = lambda self, other: self._binary_op(other, np.multiply)
    __rmul__ = __mul__
    __add__ = lambda self, other: self._binary_op(other, np.add)
    __radd__ = __add__
    __sub__ = lambda self, other: self._binary_op(other, np.subtract)
    __rsub__ = lambda self, other: np.subtract(other, np.asarray(self.value))
    __truediv__ = lambda self, other: self._binary_op(other, np.divide)
    __rtruediv__ = lambda self, other: np.divide(other, np.asarray(self.value))
    __neg__ = lambda self: np.negative(np.asarray(self.value))


def sourceH00(
    r,
    p0,
    rho0,
    drho0,
    d2rho0,
    h00,
    dh00,
    lamb,
    nu,
    eta,
    deta,
    d2eta,
):
    """
    Evaluate the L=2 source term for h00.

    All inputs should be NumPy arrays (or scalars) of matching shape:
    - r: radial coordinate
    - p0: background pressure p0(r)
    - rho0: background density ρ0(r)
    - drho0: dρ0/dr
    - d2rho0: d²ρ0/dr²
    - h00: h00(r)
    - dh00: dh00/dr
    - lamb: λ(r)
    - nu: ν(r)
    - eta: η(ρ0(r))
    - deta: dη/dρ0 evaluated at ρ0(r)
    - d2eta: d²η/dρ0² evaluated at ρ0(r)
    """
    fn = _get_lambdified_source()

    # Wrap non-callable eta/deta/d2eta so they can be used both as functions of rho0
    # and as plain numeric factors inside the expression.
    def _wrap(x):
        return x if callable(x) else _CallableArray(x)

    eta_obj = _wrap(eta)
    deta_obj = _wrap(deta)
    d2eta_obj = _wrap(d2eta)

    return fn(r, p0, rho0, drho0, d2rho0, h00, dh00, lamb, nu, eta_obj, deta_obj, d2eta_obj)


def sourceH_new(
    r,
    p0,
    rho0,
    drho0,
    d2rho0,
    h00,
    dh00,
    lamb,
    nu,
    eta,
    deta,
    d2eta,
):
    """
    Alias for sourceH00 with the same wrapping behavior for eta/deta/d2eta.
    """
    return sourceH00(r, p0, rho0, drho0, d2rho0, h00, dh00, lamb, nu, eta, deta, d2eta)

# Keep a convenient handle for the primary function name

