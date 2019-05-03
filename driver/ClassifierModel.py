from driver.Layer import *
from torch.autograd import Function



class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        reverse_grad_output = grad_output.neg()
        return reverse_grad_output

class ClassifierModel(nn.Module):
    def __init__(self, config):
        super(ClassifierModel, self).__init__()
        self.Linear = nn.Linear(config.lstm_hiddens * 2, config.lstm_hiddens, True)

        self.MLP = NonLinear(
            input_size = config.lstm_hiddens,
            hidden_size = config.lstm_hiddens,
            activation = nn.LeakyReLU(0.1))
        self.output = nn.Linear(config.lstm_hiddens, 2, False)

    def forward(self, lstm_hidden, masks):

        hidden = self.avg_pooling(lstm_hidden, masks)
        hidden = self.Linear.forward(hidden)
        hidden = ReverseLayerF.apply(hidden)
        mlp_hidden = self.MLP(hidden)
        score = self.output(mlp_hidden)
        return score

    def avg_pooling(self, lstm_hidden, masks):
        sum_hidden = torch.sum(lstm_hidden, 1)
        len = torch.sum(masks, -1, keepdim=True)
        hidden = sum_hidden / len
        return hidden




