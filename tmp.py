import numpy as np

from gt4py import gtscript


FloatFieldIJ = gtscript.Field[gtscript.IJ, np.float64]

OMEGA = 1.0


f = np.ran


@gtscript.stencil(backend="gtc:dace", rebuild=True)
def compute_coriolis_parameter_defn(
    f: FloatFieldIJ, lon: FloatFieldIJ, lat: FloatFieldIJ, alpha: float
):
    with computation(FORWARD), interval(-1, None):
        f = 2.0 * OMEGA * (-1.0 * cos(lon) * cos(lat) * sin(alpha) + sin(lat) * cos(alpha))
