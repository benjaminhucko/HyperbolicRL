import flax.linen as nn

class CNN(nn.Module):
    output_features: int
    hidden_features: int = 64
    kernel_size: int = 3
    stride: int = 1
    num_layers: int = 3


    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.Conv(features=self.hidden_features,
                        kernel_size=(self.kernel_size, self.kernel_size),
                        strides=(self.stride, self.stride),
                        padding='VALID')(x)
            x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_features)(x)
        return x