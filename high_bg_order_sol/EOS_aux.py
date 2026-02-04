# EOS_aux.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


# This is the base class, all the new equation of states inherits this class
class EOS(ABC):
    """
    Base class: define the interface your solvers rely on. This base class should be relatively light weight.
    """
    @abstractmethod
    def energy_density(self, p: float) -> float:
        """Return epsilon(p)."""
        raise NotImplementedError
    
    # Optional but often useful
    def rest_mass_density(self, p: float) -> float:
        """Return rho0(p) if available."""
        raise NotImplementedError

    def __call__(self, p: float) -> float:
        # convenience: eos(p) == eos.energy_density(p)
        return self.energy_density(p)
    

@dataclass(frozen=True)
class PolytropeEOS(EOS):
    """
    Your current two mappings packaged as one EOS object.
    You can rename to match your conventions.
    """
    n: float
    K: float

    def energy_density(self, p: float) -> float:
        # your EnergyEOS expression
        K = self.K
        return ((1.0 + np.sqrt(4.0 * K * p))**2 - 1.0) / (4.0 * K)
    
    def rest_mass_density(self, p: float) -> float:
        # rho0(p) = (p/K)^(n/(n+1))
        return (p / self.K) ** (self.n / (self.n + 1.0))
