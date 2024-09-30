# This is the gravity model parent class
class ModelBC:
    def __init__(self):
        pass

    # This method computes gravity acceleration
    def compute_acc(self, pos):
        pass

    # This method computes gravity potential
    def compute_U(self, pos):
        pass
