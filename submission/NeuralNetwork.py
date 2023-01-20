#  Team-Mitglieder : Huang, Jin [an46ykim]; Liu, Peicheng [ha46tika]
import copy


class NeuralNetwork:

    def __init__(self, optimizer):
        self.optimizer = None
        self.loss = list()
        self.layers = list()
        self.data_layer = None
        self.loss_layer = None

        self.input_tensor = None
        self.label_tensor = None
        self.optimizer = optimizer

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        output_temp = self.input_tensor
        for i in range(len(self.layers)):
            output_temp = self.layers[i].forward(output_temp)
        output_temp = self.loss_layer.forward(output_temp, self.label_tensor)
        self.loss.append(output_temp)
        return output_temp

    def backward(self):
        error_temp = self.loss_layer.backward(self.label_tensor)
        for i in reversed(range(len(self.layers))):
            error_temp = self.layers[i].backward(error_temp)
        return error_temp

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            self.forward()
            self.backward()

    def test(self, input_tensor):
        output_temp = input_tensor
        for i in range(len(self.layers)):
            output_temp = self.layers[i].forward(output_temp)
        return output_temp