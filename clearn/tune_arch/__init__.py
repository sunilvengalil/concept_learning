class Architecture:
    def __init__(self, num_layers, strides, z_dim):
        self.strides = strides
        self.z_dim = z_dim
        self.num_units_list =[[512, 128, 64, 64, 64],
                         [512, 128, 64, 64, 32],
                         [512, 128, 64, 32, 32],
                         [512, 128, 32, 32, 32],
                         [512, 64, 32, 32, 32],
                         [256, 128, 64, 64, 64],
                         [256, 128, 64, 64, 32],
                         [512, 128, 64, 32, 32],
                          [512, 128, 32, 32, 32],
                          [512, 64, 64, 64, 64],
                          [512, 64, 64, 64, 32],
                          [512, 64, 64, 32, 32],
                          [512, 64, 32, 32, 32],
                          [512, 32, 32, 32, 32]
                         ][1:]
        self.num_layers = len(self.num_units_list[0])
        self.index = 0

    def reset_index(self):
        self.index = 0

    def get_next(self):
        if self.index == len(self.num_units_list):
            raise Exception("Last item reached. Call reset index to restart")
        self.index += 1
        return self.z_dim, self.strides, self.num_units_list[self.index - 1]