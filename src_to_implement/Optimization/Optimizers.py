#  Team-Mitglieder : Huang, Jin [an46ykim]; Liu, Peicheng [ha46tika]
class Sgd:
    # 更新参数，权重 update weights
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = weight_tensor - self.learning_rate * gradient_tensor
        return weight_tensor
