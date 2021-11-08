"""Settings and configuration that apply across the whole project


Sentinel 2 bands
----------------

https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/resolutions/radiometric

=============   =========== ======================  ======================  ====================
 Band label     Band code   Band description        Central wavelength      Resolution (m)
=============   =========== ======================  ======================  ====================
Band 1          "B01"       Coastal aerosol	        0.443	                 60
Band 2          "B02"       Blue	                0.49	                 10
Band 3          "B03"       Green                   0.56	                 10
Band 4          "B04"       Red                     0.665	                 10
Band 5          "B05"       Vegetation Red Edge     0.705	                 20
Band 6          "B06"       Vegetation Red Edge     0.74	                 20
Band 7          "B07"       Vegetation Red Edge     0.783	                 20
Band 8          "B08"       NIR	                    0.842	                 10
Band 8A         "B8a"       Vegetation Red Edge     0.865	                 20
Band 9          "B09"       Water vapour	        0.945	                 60
Band 10         "B10"       SWIR - Cirrus	        1.375	                 60
Band 11         "B11"       SWIR	                1.61	                 20
Band 12         "B12"       SWIR	                2.19	                 20
=============   =========== ======================  ======================  ====================

> The `Band code` is used to refer to Sentinel-2 bands throughout the project.

"""

DEFAULT_BAND_SEQUENCE = (
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B8a",
    "B11",
    "B12",
    "COS_VIEW_ZENITH",
    "COS_SUN_ZENITH",
    "COS_REL_AZIMUTH"
)
"""Default sequence of Sentinel-2 bands & meta-data assumed throughout project"""
