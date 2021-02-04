import torch
from torch.nn import Linear, Conv2d, Module, Sequential, BatchNorm2d, Linear
from torch.nn.functional import relu, softplus, max_pool2d


class ConvBlock(Module):
    def __init__(self, in_channels, out_channels, maml=False, **kwargs):
        super(ConvBlock, self).__init__()
        self.main_net = Conv2d(in_channels, out_channels, **kwargs)
        self.noise_net = Conv2d(in_channels, out_channels, **kwargs)
        self.batch_norm = BatchNorm2d(out_channels)
        self.maml = maml

    def forward(self, inputs, params=None):

        theta_x = self.main_net(inputs)
        phi_x = self.noise_net(inputs)
        # var = torch.ones_like(phi_x)
        if self.maml:
            f = theta_x
        else:
            # f = torch.distributions.Normal(
            #     phi_x, var).rsample(sample_shape=phi_x.size())
            f = phi_x + torch.randn(phi_x.size())
            f = theta_x * softplus(f)

        f = self.batch_norm(f)
        f = relu(f)
        f = max_pool2d(f, kernel_size=2, stride=2)

        return f


class MetaDropout(Module):
    def __init__(self, args):
        super(MetaDropout, self).__init__()
        self.hidden_size = 64 if args.dataset == 'omniglot' else 32
        self.input_channels = 1 if args.dataset == 'omniglot' else 3
        self.k_shot = args.num_shots
        self.n_way = args.num_ways
        self.features = Sequential(*[
            ConvBlock(self.input_channels if i == 0 else self.hidden_size,
                      self.hidden_size,
                      args.maml,
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      bias=True) for i in range(4)
        ])
        self.feature_size = self.hidden_size if args.dataset == 'omniglot' else self.hidden_size * 5 * 5
        self.dense_layer_main = Linear(self.feature_size, self.n_way)
        self.dense_layer_noise = Linear(self.feature_size, self.n_way)

    def get_main_net_params(self):

        for name, params in self.named_parameters():
            if name.find('noise') == -1:
                yield params

    def get_other_params(self):

        for name, params in self.named_parameters():
            if name.find('noise') != -1:
                yield params

    def forward(self, inputs):

        features = self.features(inputs)
        features = features.view((features.size(0), -1))
        # Additive noise
        # mu = self.dense_layer_main(features,
        #                            params=self.get_subdict(
        #                                params, 'dense_layer_main'))
        mu = self.dense_layer_main(features)
        sigma = softplus(self.dense_layer_noise(features))
        # logits = torch.distributions.Normal(
        #     mu, sigma).rsample(sample_shape=mu.size())
        logits = mu + sigma * torch.randn(mu.size())
        return logits
