import ctypes
import glob
import os
import sys


# Preload CUDA libraries bundled in virtual environments if present so dynamic
# linking succeeds without requiring manual LD_LIBRARY_PATH exports.
_candidate_dirs = [
    os.path.join(
        sys.prefix,
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages",
        "nvidia",
        "cu13",
        "lib",
    ),
    os.environ.get("CUDA_CU13_LIB", ""),
]

for _cu13_dir in _candidate_dirs:
    if _cu13_dir and os.path.isdir(_cu13_dir):
        if _cu13_dir not in os.environ.get("LD_LIBRARY_PATH", ""):
            os.environ["LD_LIBRARY_PATH"] = (
                _cu13_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
            )
        for _p in sorted(glob.glob(os.path.join(_cu13_dir, "*.so*"))):
            try:
                ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass
        break

