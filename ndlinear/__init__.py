from ndlinear.modules.ndlinear_gated import NdLinearGated

import importlib.util
import os
import sys

spec = importlib.util.spec_from_file_location("ndlinear_module", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ndlinear.py"))
ndlinear_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ndlinear_module)
NdLinear = ndlinear_module.NdLinear

__all__ = ['NdLinear', 'NdLinearGated']
