import numpy as np
import bruges


class SyntheticCube():
    '''
    Building and saving synthetic seismic cubes.
    Able to create models with layering of different thicknesses, inclination and folding.
    '''

    def __init__(self, size=[256, 256, 256]):
        self.size = size
        self.reflectivity = np.random.normal(0, 0.01, self.size)
        self.horizons = np.zeros(size)
        self.seismic = None
        self.layers_depths = None
        self.layers_impedances = None

    def create_horizontal_reflectivity(self, max_reflectivity=0.3):
        '''
        General function for creating starting cube of horizontal layers.
        Reflectivity coefficients and thicknesses samples randomly from uniform distributions.
        '''
        self.layers_depths = []
        self.layers_impedances = []
        inter_thick = 0
        while inter_thick < self.size[0]:
            random_thickness = np.random.randint(5, 15)
            random_reflectivity = np.random.uniform(
                -max_reflectivity, max_reflectivity)
            inter_thick += random_thickness
            self.layers_depths.append(inter_thick)
            self.layers_impedances.append(random_reflectivity)

        for idx, i in enumerate(self.layers_depths[:-1]):
            self.reflectivity[i, :, :] = self.layers_impedances[idx]
            self.horizons[i, :, :] = idx+1

    # def add_amplitude_anomalies(self, probability=0.05):
    #     for hor in range(np.unique(self.horizons).size):
    #         rand = np.random.rand()
    #         if rand < probability:
    #             anomaly_half_width = np.random.randint(
    #                 self.size[0]//15, self.size[0]//10)
    #             place = np.random.randint(
    #                 anomaly_half_width, self.size[0]-anomaly_half_width)
    #             a, b = place-anomaly_half_width, place+anomaly_half_width
    #             self.reflectivity[:, :,
    #                            a:b][self.horizons[:, :, a:b] == hor] *= 2

    def __fold_layer(self, n_structures=10, amp=50, power=10000):
        z_start = np.zeros(self.size[:2])
        x = np.arange(0, self.size[0])
        y = np.arange(0, self.size[0])
        for i in range(n_structures):
            x_center = np.random.randint(0, self.size[0])
            y_center = np.random.randint(0, self.size[1])
            sigma_x = np.random.randint(
                int(self.size[0]/5), int(self.size[0]/2))
            sigma_y = np.random.randint(
                int(self.size[0]/5), int(self.size[1]/2))
            amplifier = np.random.normal(0, amp)
            x_m, y_m = np.meshgrid(x, y)
            same_part = amplifier*power*(1/(2*np.pi*sigma_x*sigma_y))
            z = same_part * \
                np.exp(-((x_m-x_center)**2/(2*sigma_x**2) +
                         (y_m-y_center)**2/(2*sigma_y**2)))
            z_start = z_start + z
        return z_start.round().astype(int)

    def __influence_thickness(self, top_change, bot_change):
        z_start = self.__fold_layer(n_structures=5, amp=15, power=10000)
        z_start[z_start > bot_change] = bot_change
        z_start[z_start < top_change] = top_change
        return z_start

    def change_layers_thickness(self, probability=0.5):
        '''
        Function for changing layers thickness.
        Allows making pinchouts and adding subresolution layers.
        probability (float): determines the chance that layers thickness will not be changed.
        '''
        for ddx, d in enumerate(self.layers_depths[1:-2]):
            rand = np.random.rand()
            if rand > probability:
                top_change = self.layers_depths[ddx] - \
                    self.layers_depths[ddx+1] + 1
                bot_change = self.layers_depths[ddx +
                                                2] - self.layers_depths[ddx+1] - 1
                thickness = self.__influence_thickness(top_change, bot_change)
                for i in range(self.reflectivity.shape[1]):
                    for j in range(self.reflectivity.shape[2]):
                        hor = self.reflectivity[d, i, j]
                        self.reflectivity[d, i, j] = np.random.normal(0, 0.01)
                        self.reflectivity[d+thickness[i, j], i, j] = hor

                        hor = self.horizons[d, i, j]
                        self.horizons[d, i, j] = 0
                        self.horizons[d+thickness[i, j], i, j] = hor

    def incline_layers(self, degree=0.3):
        '''
        Function for inclination of the whole cube volume.
        degree (float): determines maximum angles of inclination
        '''
        a, b = np.random.uniform(-degree,
                                 degree), np.random.uniform(-degree, degree)
        c = -a*(self.size[0]//2) - b*(self.size[0]//2)
        for i in range(self.reflectivity.shape[1]):
            for j in range(self.reflectivity.shape[2]):
                shift = a*i + b*j + c
                self.reflectivity[:, i, j] = np.roll(
                    self.reflectivity[:, i, j], round(shift))
                self.horizons[:, i, j] = np.roll(
                    self.horizons[:, i, j], round(shift))

    def folding_all_horizons(self):
        '''
        Function for folding of the whole cube volume.
        '''
        fold_model = self.__fold_layer(n_structures=10, amp=50, power=20000)
        for i in range(self.reflectivity.shape[1]):
            for j in range(self.reflectivity.shape[2]):
                self.reflectivity[:, i, j] = np.roll(
                    self.reflectivity[:, i, j], fold_model[i, j])
                self.horizons[:, i, j] = np.roll(
                    self.horizons[:, i, j], fold_model[i, j])

    def convolve_impulse(self):
        '''
        Convolution of reflectivity coefficients cube with Rickers seismic impulse.
        '''
        impulses = [bruges.filters.ricker(duration=0.098, dt=0.002, f=fr)
                    for fr in np.arange(5, 51, 5)]
        impulses_amp = np.linspace(0, 1, 3).tolist() + \
            [1]*4 + np.linspace(1, 0.2, 3).tolist()
        impulses = [impulses[i]*impulses_amp[i]
                    for i in range(len(impulses_amp))]
        synthetic_diff_freq = [np.apply_along_axis(
            lambda t: np.convolve(t, impulse), axis=0, arr=self.reflectivity) for impulse in impulses]
        self.seismic = np.sum(synthetic_diff_freq, axis=0)[24:-24]

    def add_gauss_noise(self, noise_amp=0.1):
        '''
        Add gaussian noise to convolved seismic cube.
        noise_amp (float): determines power of noise.
        '''
        self.seismic += np.random.rand(*self.reflectivity.shape)*noise_amp

    def __create_cube(self):
        self.create_horizontal_reflectivity()
        # self.add_amplitude_anomalies()
        self.change_layers_thickness()
        self.incline_layers()
        self.folding_all_horizons()
        self.convolve_impulse()
        self.add_gauss_noise()

    def build_seismic(self, binarize_horizons=True):
        '''
        Function builds cubes of reflectivity coefficients, horizons and seismic.
        Args:
        binarize_horizons (bool, float): If True - horizons will be binarized (is horizon or not)
        If float horizons with absolute reflectivity higher than float will be left binarized
        and only them. Default: True

        Returns: np.arrays of seismic cube, horizon cube and impedance cube
        '''
        self.__create_cube()
        if binarize_horizons and type(binarize_horizons) == bool:
            self.horizons[self.horizons != 0] = 1
        elif type(binarize_horizons) == float:
            self.horizons[abs(self.reflectivity) >= binarize_horizons] = 1
            self.horizons[abs(self.reflectivity) < binarize_horizons] = 0
        return self.seismic, self.horizons, self.reflectivity

    def save_cube(self,
                  seismic_dir_name: str,
                  horizons_dir_name: str,
                  reflectivity_dir_name: str,
                  binarize_horizons=True):
        '''
        Function builds and saves numpy arrays of  three cubes:
        reflectivity coefficients, horizons and seismic.

        Args:
        seismic_dir_name (str): path for seismic file
        horizons_dir_name (str): path for horizons file
        reflectivity_dir_name (str): path for impedance file
        '''
        if self.seismic is None:
            self.build_seismic(binarize_horizons)

        np.save(seismic_dir_name, self.seismic)
        np.save(horizons_dir_name, self.horizons)
        np.save(impedance_dir_name, self.reflectivity)
