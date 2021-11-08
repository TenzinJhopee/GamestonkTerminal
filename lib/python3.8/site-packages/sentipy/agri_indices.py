"""
There are a wide range of agriculturally-relevant indices to be processed from (multi-spectral) optical satellite imagery in general.

This module provides implementations of a number of indices, largely inspired by the `Index Database <https://www.indexdatabase.de>`_ and the `SentinelHub custom scripts repository <https://custom-scripts.sentinel-hub.com/custom-scripts/>`_.
They are implemented principally with Sentinel-2 in mind, but can be applied to imagery from other constellations where equivalent bands are available (it is left to the user to decide where this is the case).

Available implementations:

- NDVI
- NDWI
- OSAVI



"""


def ndvi(img_arr, band_sequence: tuple = ("B04", "B08")):
    """Normalized Difference Vegetation Index

    NDVI    = (NIR - RED)/(NIR + RED)
            = (B08 - B04)/ (B08 + B04)

    """
    red_idx = band_sequence.index("B04")
    nir_idx = band_sequence.index("B08")
    red = img_arr[:,:,red_idx]
    nir = img_arr[:,:,nir_idx]
    return ((nir - red)/(nir + red))

def ndwi(img_arr, band_sequence: tuple = ("B03", "B08")):
    """Normalized Difference Water Index

    NDWI    = -(GREEN - NIR)/(GREEN + NIR)
            = -(B03 - B08)/(B03 + B08)
    """
    green_idx = band_sequence.index("B03")
    nir_idx = band_sequence.index("B08")
    green = img_arr[:,:,green_idx]
    nir = img_arr[:,:,nir_idx]
    return (-(green - nir)/(green + nir))

def ndmi(img_arr, band_sequence: tuple = ("B08", "B11")):
    """Normalized Difference Moisture Index

    NDMI  = (NIR - SWIR)/(NIR + SWIR)
          = (B08 - B11)/(B08 + B11)
    """
    nir_idx = band_sequence.index("B08")
    swir_idx = band_sequence.index("B11")
    nir = img_arr[:,:,nir_idx]
    swir = img_arr[:,:,swir_idx]
    return ((nir - swir)/(nir + swir))


def osavi(img_arr, band_sequence: tuple = ("B04", "B08")):
    """Optimized Soil-Adjusted Vegetation Index
    `Index DB link <https://www.indexdatabase.de/db/i-single.php?id=63>`_


    OSAVI   = (1+0.16)*(NIR - RED)/(NIR + RED + 0.16)
            = (1.16)*(B08 - B04)/(B08 + B04 + 0.16)

    """
    red_idx = band_sequence.index("B04")
    nir_idx = band_sequence.index("B08")
    red = img_arr[:,:,red_idx]
    nir = img_arr[:,:,nir_idx]
    return (1.16*(nir - red)/(nir + red + 0.16))

# def seli(self):
#     pass