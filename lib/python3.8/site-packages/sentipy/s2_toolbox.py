"""Home for the objects & logic that deal with the calculation of FAPAR
"""
from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np

from sentipy.lib.neuralnet import Neuron, Network
from sentipy.lib.preprocessing import Normaliser
from sentipy.settings import DEFAULT_BAND_SEQUENCE


class S2BiophysicalCalculator(ABC):

    def __init__(self):
        """Calculates BioPhysical property values from Sentinel-2 imagery as per the S2 toolbox products
        """
        self._initialise_normalisation()
        self._initialise_network()

    def _initialise_normalisation(self):
        """Initialise the Normalisers to preprocess inputs according to the expected/validated minimums & maximums"""
        self.norm_b3 = Normaliser(x_min=0., x_max=0.253061520471542)
        self.norm_b4 = Normaliser(x_min=0., x_max=0.290393577911328)
        self.norm_b5 = Normaliser(x_min=0., x_max=0.305398915248555)
        self.norm_b6 = Normaliser(x_min=0.006637972542253, x_max=0.608900395797889)
        self.norm_b7 = Normaliser(x_min=0.013972727018939, x_max=0.753827384322927)
        self.norm_b8a = Normaliser(x_min=0.026690138082061, x_max=0.782011770669178)
        self.norm_b11 = Normaliser(x_min=0.016388074192258, x_max=0.493761397883092)
        self.norm_b12 = Normaliser(x_min=0., x_max=0.493025984460231)
        self.norm_cos_view_zenith = Normaliser(x_min=0.918595400582046, x_max=0.99999999999139)
        self.norm_cos_sun_zenith = Normaliser(x_min=0.342022871159208, x_max=0.936206429175402)
        self.norm_cos_rel_azimuth = Normaliser(x_min=-0.999999982118044, x_max=0.999999998910077)
        self._initialise_output_normalisation()

    @abstractmethod
    def _initialise_output_normalisation(self):
        """Define the normalisation parameters on the output to return outputs into the natural range of values

        MUST be overridden in concrete child classes.
        """
        self.norm_output = None

    @abstractmethod
    def _initialise_network(self):
        """Define the network structure & parameterisation

        MUST be overridden in concrete child classes
        """
        self.neuron_1 = None
        self.neuron_2 = None
        self.neuron_3 = None
        self.neuron_4 = None
        self.neuron_5 = None
        self.neuron_6 = None
        self.network = None

    def run(self, input_arr: np.ndarray, band_sequence: List[str] = DEFAULT_BAND_SEQUENCE, validate: bool = True) -> \
            Union[float, np.float]:
        """Run the calculator on an input array

        By default, the calculator expects only the following bands to be passed in the sequence below:

        - B03
        - B04
        - B05
        - B06
        - B07
        - B8a
        - B11
        - B12
        - COS_VIEW_ZENITH
        - COS_SUN_ZENITH
        - COS_REL_AZIMUTH

        If band values are to be passed in a different set or sequence, the band_sequence parameter must be passed with
        band names (matching those above) for each element in the input array.
        eg. ["extra_band_1", "COS_SUN_ZENITH", "B03", "B04", ..., "COS_REL_AZIMUTH", "extra_band_2"]

        :param input_arr: Input values for the calculator to use
        :param band_sequence: Names of bands included in the input array (names must match those used above for the required bands)
        :param validate: Flag for whether or not to apply validation ranges to the inputs
        :return: Scalar estimate of the biophysical property
        """
        if not band_sequence == DEFAULT_BAND_SEQUENCE:
            band_idxs = [band_sequence.index(elem) for elem in DEFAULT_BAND_SEQUENCE]
            ordered_arr = np.array([input_arr[idx] for idx in band_idxs])
        else:
            ordered_arr = input_arr
        if validate:
            ordered_arr = self._validate(ordered_arr)
        normalised_arr = self._normalise(ordered_arr)
        y_norm = self._compute(normalised_arr)
        y = self.norm_output.denormalise(y_norm)
        return y

    def _validate(self, input_arr: np.ndarray) -> np.ndarray:
        """Validate input band values against the 'definition domain for inputs'

        NB. We validate against the min & max input values, but we do NOT check that inputs lie in valid cells of the
        convex hull

        :param input_arr: Band values to be validated
        :return: Band values after passing through validation. ValueError is raised if any values fail.
        """
        validation_ranges = self.VALIDATION_RANGES
        for band_index, band_name in enumerate(validation_ranges):
            band_value = input_arr[band_index]
            if validation_ranges.get(band_name).get("min") <= band_value <= validation_ranges.get(band_name).get("max"):
                continue
            else:
                raise ValueError(
                    f"Band {band_name} failed validation because it is expected to fall in the range [{validation_ranges.get(band_name).get('min')}, {validation_ranges.get(band_name).get('max')}]")
        return input_arr

    def _normalise(self, band_values: np.ndarray) -> np.ndarray:
        """Normalise input band values to predefined ranges before passing to the neural network for calculation.

        :param band_values: Band values to be normalised
        :return: Normalised band values
        """
        return np.array([
            self.norm_b3.normalise(band_values[0]),
            self.norm_b4.normalise(band_values[1]),
            self.norm_b5.normalise(band_values[2]),
            self.norm_b6.normalise(band_values[3]),
            self.norm_b7.normalise(band_values[4]),
            self.norm_b8a.normalise(band_values[5]),
            self.norm_b11.normalise(band_values[6]),
            self.norm_b12.normalise(band_values[7]),
            self.norm_cos_view_zenith.normalise(band_values[8]),
            self.norm_cos_sun_zenith.normalise(band_values[9]),
            self.norm_cos_rel_azimuth.normalise(band_values[10]),
        ])

    def _compute(self, normalised_arr: np.ndarray) -> np.float:
        """Calculates normalised FAPAR from normalised input band values

        :param normalised_arr: Normalised band values
        :return: Normalised FAPAR estimate
        """
        return self.network.forward(normalised_arr)


class Fapar(S2BiophysicalCalculator):
    """Calculator for the Fapar biophysical parameter; the fraction of photosynthetically-active radiation absorbed (by vegetation)."""

    VALIDATION_RANGES = {
        "B03": {
            "min": 0.,
            "max": 0.263
        },
        "B04": {
            "min": 0.,
            "max": 0.300
        },
        "B05": {
            "min": 0.,
            "max": 0.315
        },
        "B06": {
            "min": 0.,
            "max": 0.619
        },
        "B07": {
            "min": 0.004,
            "max": 0.764
        },
        "B8a": {
            "min": 0.017,
            "max": 0.792
        },
        "B11": {
            "min": 0.006,
            "max": 0.503
        },
        "B12": {
            "min": 0.,
            "max": 0.503
        },
        "Cos(view zenith)": {
            "min": 0.,
            "max": 1.
        },
        "Cos(sun zenith)": {
            "min": 0.,
            "max": 1.
        },
        "Cos(rel. azimuth)": {
            "min": -1.,
            "max": 1.
        },
    }

    def _initialise_network(self):
        self.neuron_1 = Neuron(
            weights=np.array([
                0.268714454733421, -0.205473108029835, 0.281765694196018, 1.33744341225598, 0.390319212938497,
                -3.61271434220335, 0.222530960987244, 0.821790549667255, -0.093664567310731, 0.019290146147447,
                0.037364446377188,
            ]),
            bias=-0.88706836404028,
            activation='tansig'
        )
        self.neuron_2 = Neuron(
            weights=np.array([
                -0.248998054599707, -0.571461305473124, -0.369957603466673, 0.246031694650909, 0.332536215252841,
                0.438269896208887, 0.81900055189045, -0.93493149905931, 0.082716247651866, -0.286978634108328,
                -0.035890968351662
            ]),
            bias=0.320126471197199,
            activation='tansig'
        )
        self.neuron_3 = Neuron(
            weights=np.array([
                -0.16406357531588, -0.126303285737763, -0.253670784366822, -0.321162835049381, 0.06708228797358,
                2.02983228865526, -0.023141228827722, -0.553176625657559, 0.059285451897783, -0.034334454541432,
                -0.031776704097009
            ]),
            bias=0.610523702500117,
            activation='tansig'
        )
        self.neuron_4 = Neuron(
            weights=np.array([
                0.130240753003835, 0.236781035723321, 0.131811664093253, -0.250181799267664, -0.011364149953286,
                -1.85757321463352, -0.146860751013916, 0.528008831372352, -0.046230769098303, -0.034509608392235,
                0.031884395036004
            ]),
            bias=-0.379156190833946,
            activation='tansig'
        )
        self.neuron_5 = Neuron(
            weights=np.array([
                -0.029929946166941, 0.795804414040809, 0.348025317624568, 0.943567007518504, -0.276341670431501,
                -2.94659418014259, 0.2894830735075, 1.04400695044018, -0.000413031960419, 0.403331114840215,
                0.068427130526696
            ]),
            bias=1.35302339669057,
            activation='tansig'
        )
        self.neuron_6 = Neuron(
            weights=np.array([
                2.12603881106449, -0.632044932794919, 5.59899578720625, 1.77044414057897, -0.267879583604849
            ]),
            bias=-0.336431283973339,
            activation='linear'
        )
        self.network = Network(
            hidden_layers=(
                [self.neuron_1, self.neuron_2, self.neuron_3, self.neuron_4, self.neuron_5],
            ),
            output_neuron=self.neuron_6
        )

    def _initialise_output_normalisation(self):
        self.norm_output = Normaliser(x_min=0.000153013463222, x_max=0.977135096979553)


class Fcover(S2BiophysicalCalculator):
    """Calculator for the Fcover biophysical parameter; the fraction of ground covered by vegetation."""

    def _initialise_network(self):
        self.neuron_1 = Neuron(
            weights=np.array([
                -0.156854264840505, 0.124234528461836, 0.235625516228529, -1.83239102580498, -0.217188969888118,
                5.06933958064326, -0.88757800815474, -1.08084681669584, -0.032316704186389, -0.224476137358619,
                -0.195523962947318
            ]),
            bias=-1.45261652205846,
            activation='tansig'
        )
        self.neuron_2 = Neuron(
            weights=np.array([
                -0.220824927841957, 1.28595395486941, 0.70313948636251, -1.34481216664598, -1.96881267558705,
                -1.45444681638688, 1.0273756004279, -0.124946415319548, 0.080276243726516, -0.198705918577447,
                0.108527100526934
            ]),
            bias=-1.70417477557288,
            activation='tansig'
        )
        self.neuron_3 = Neuron(
            weights=np.array([
                -0.409688743280695, 1.08858884765636, 0.362845225540078, 0.036939050970548, -0.348012590003251,
                -2.00352618809814, 0.041035760175655, 1.22373853174142, -0.012408277828702, -0.282223364523503,
                0.099499311755661
            ]),
            bias=1.02168965848613,
            activation='tansig'
        )
        self.neuron_4 = Neuron(
            weights=np.array([
                -0.188970957866161, -0.035862184083284, 0.005512485281073, 1.35391570802373, -0.739689896116339,
                -2.21719530107254, 0.313216124198161, 1.50201689149522, 1.21530490194501, -0.421938358618199,
                1.48852484546637
            ]),
            bias=-0.49800281020533,
            activation='tansig'
        )
        self.neuron_5 = Neuron(
            weights=np.array([
                2.4929399370874, -4.40511331388413, -1.91062012624287, -0.703174115574677, -0.215104721137593,
                -0.972151494817506, -0.930752241278312, 1.21434418759821, -0.521665460191844, -0.445755955597775,
                0.344111873776809
            ]),
            bias=-3.88922154789449,
            activation='tansig'
        )
        self.neuron_6 = Neuron(
            weights=np.array([
                0.230805867649849, -0.333655484884161, -0.499418292324876, 0.047248439674868, -0.079851654073939
            ]),
            bias=-0.096799814781108,
            activation='linear'
        )
        self.network = Network(
            hidden_layers=(
                [self.neuron_1, self.neuron_2, self.neuron_3, self.neuron_4, self.neuron_5],
            ),
            output_neuron=self.neuron_6
        )

    def _initialise_output_normalisation(self):
        self.norm_output = Normaliser(x_min=0.000181230723879, x_max=0.999638214714515)

    def _validate(self, input_arr: np.ndarray) -> np.ndarray:
        """Validate input band values against the 'definition domain for inputs'

        NB. There is no definition domain for inputs available in the FCOVER data, so no validation is applied here

        :param input_arr: Band values to be validated
        :return: Band values after passing through validation. ValueError is raised if any values fail.
        """
        return input_arr


class CanopyWater(S2BiophysicalCalculator):
    """Calculator for the Canopy Water biophysical parameter."""

    VALIDATION_RANGES = {
        "B03": {
            "min": 0.,
            "max": 0.263
        },
        "B04": {
            "min": 0.,
            "max": 0.300
        },
        "B05": {
            "min": 0.,
            "max": 0.315
        },
        "B06": {
            "min": 0.,
            "max": 0.619
        },
        "B07": {
            "min": 0.004,
            "max": 0.764
        },
        "B8a": {
            "min": 0.017,
            "max": 0.792
        },
        "B11": {
            "min": 0.006,
            "max": 0.503
        },
        "B12": {
            "min": 0.,
            "max": 0.503
        },
        "Cos(view zenith)": {
            "min": 0.,
            "max": 1.
        },
        "Cos(sun zenith)": {
            "min": 0.,
            "max": 1.
        },
        "Cos(rel. azimuth)": {
            "min": -1.,
            "max": 1.
        },
    }

    def _initialise_output_normalisation(self):
        self.norm_output = Normaliser(x_min=0.00000385066859365878, x_max=0.522417054644758)

    def _initialise_network(self):
        self.neuron_1 = Neuron(
            weights=np.array([
                0.146378710426108, 1.18979928186603, -0.906235139963469, -0.808337508767272, -0.973334917830317,
                -1.42591277645983, -0.005612536295883, -0.634520356266989, -0.117226059988763, -0.060270091210205,
                0.229407587131601

            ]),
            bias=-2.10640836859515,
            activation='tansig'
        )
        self.neuron_2 = Neuron(
            weights=np.array([
                0.283319173374232, 0.14934202304068, 1.08480588386826, -0.138658791034905, -0.455759407328658,
                0.420571438077985, -1.73729490369704, -0.704286287225911, 0.01909537823579, -0.039397131651267,
                -0.007502415817443
            ]),
            bias=-1.69022094794167,
            activation='tansig'
        )
        self.neuron_3 = Neuron(
            weights=np.array([
                -0.197487427943115, -0.105460325978344, 0.158347670680985, 2.14912426653839, -0.970716842915524,
                -4.92725317908744, 1.42034301781109, 1.45316917225712, 0.022725705360916, 0.269298650421124,
                0.084904765771522
            ]),
            bias=3.10117655254553,
            activation='tansig'
        )
        self.neuron_4 = Neuron(
            weights=np.array([
                0.141405799762781, 0.333862603279641, 0.356218929122853, -0.545942267638804, 0.089104307685555,
                0.919298362928991, -1.85208926250394, -0.427539590778633, 0.007913856464675, 0.014833320147784,
                -0.001537867697355
            ]),
            bias=-1.31231626495557,
            activation='tansig'
        )
        self.neuron_5 = Neuron(
            weights=np.array([
                -0.186781083395016, -0.549163704900838, -0.181287638772104, 0.968640436559575, -0.470442559117241,
                -1.24859725243601, 2.67014942338173, 0.490090624379901, -0.001449319395262, 0.003148293696923,
                0.020651788389291
            ]),
            bias=1.01131930348399,
            activation='tansig'
        )
        self.neuron_6 = Neuron(
            weights=np.array([
                -0.077555589034682, -0.86411786118988, -0.199212415373717, 1.98730461218687, 0.458926743488878
            ]),
            bias=-0.197591709976582,
            activation='linear'
        )
        self.network = Network(
            hidden_layers=(
                [self.neuron_1, self.neuron_2, self.neuron_3, self.neuron_4, self.neuron_5],
            ),
            output_neuron=self.neuron_6
        )
