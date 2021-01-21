import torch
from torch.nn import Linear, Conv2d
from torch.nn.functional import relu, softplus, max_pool2d
from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)
from collections import OrderedDict


class ConvBlock(MetaModule):
    def __init__(self, in_channels, out_channels, maml=False, **kwargs):
        super(ConvBlock, self).__init__()
        self.main_net = MetaConv2d(in_channels, out_channels, **kwargs)
        self.noise_net = Conv2d(in_channels, out_channels, **kwargs)
        self.batch_norm = MetaBatchNorm2d(out_channels)
        self.maml = maml

    def forward(self, inputs, params=None):

        theta_x = self.main_net(inputs,
                                params=self.get_subdict(params, 'main_net'))
        phi_x = self.noise_net(inputs)
        var = torch.ones_like(phi_x)
        if self.maml:
            f = theta_x
        else:
            f = torch.distributions.Normal(
                phi_x, var).rsample(sample_shape=phi_x.size())
            f = theta_x * softplus(f)

        f = self.batch_norm(f)
        f = relu(f)
        f = max_pool2d(f, kernel_size=2, stride=2)

        return f


class MetaDropout(MetaModule):
    def __init__(self, args):
        super(MetaDropout, self).__init__()
        self.hidden_size = 64 if args.dataset == 'omniglot' else 32
        self.input_channels = 1 if args.dataset == 'omniglot' else 3
        self.k_shot = args.num_shots
        self.n_way = args.num_ways
        self.features = MetaSequential(*[
            ConvBlock(self.input_channels if i == 0 else self.hidden_size,
                      self.hidden_size,
                      args.maml,
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      bias=True) for i in range(4)
        ])
        self.feature_size = self.hidden_size if args.dataset == 'omniglot' else self.hidden_size * 5 * 5
        self.dense_layer_main = MetaLinear(self.feature_size, self.n_way)
        self.dense_layer_noise = Linear(self.feature_size, self.n_way)

    def get_main_net_params(self):

        main_net_params = OrderedDict()
        for name, params in self.meta_named_parameters():
            if name.find('noise') == -1:
                main_net_params[name] = params

        return main_net_params

    def forward(self, inputs, params=None):

        features = self.features(inputs,
                                 params=self.get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        # Additive noise
        mu = self.dense_layer_main(features,
                                   params=self.get_subdict(
                                       params, 'dense_layer_main'))
        sigma = softplus(self.dense_layer_noise(features))
        logits = torch.distributions.Normal(
            mu, sigma).rsample(sample_shape=mu.size())
        return logits
