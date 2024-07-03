from Basilisk import __path__

bsk_path = __path__[0]


# Gravity estimation attributes
class Shape:
    def __init__(self):
        pass

    # Initializes optimizer
    def prepare_optimizer(self):
        pass

    # Checks if a point is exterior
    def check_exterior(self, pos):
        pass

    # Compute altitude w.r.t. surface
    def compute_altitude(self, pos):
        pass

    # Deletes shape
    def delete_shape(self):
        pass
