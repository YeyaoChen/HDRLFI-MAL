import torch
from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()

        vgg_pretrained_features = models.vgg19(pretrained=True).features
        # print(vgg_pretrained_features)

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 14):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = False
            for param in self.slice2.parameters():
                param.requires_grad = False
            for param in self.slice3.parameters():
                param.requires_grad = False

    def forward(self, in_x):
        # input [b,c,h,w]
        out_x = self.slice1(in_x)
        out_x = self.slice2(out_x)
        out_x = self.slice3(out_x)
        h_relu3_2 = out_x
        return h_relu3_2


if __name__ == '__main__':
    vgg19 = Vgg19(requires_grad=False)




