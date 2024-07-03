# This class encodes a gravity model
class Gravity:
    def __init__(self):
        pass

    # This method initializes optimizer
    def prepare_optimizer(self):
        pass

    # This method trains gravity
    def train(self, pos_data, acc_data):
        pass

    # This method deletes optimizer
    def delete_optimizer(self):
        pass

    # This method creates gravity evaluator
    def create_gravity(self):
        pass

    # This method computes gravity acceleration
    def compute_acc(self, pos):
        pass

    # This method computes gravity potential
    def compute_U(self, pos):
        pass

    # This method deletes gravity evaluator
    def delete_gravity(self):
        pass
