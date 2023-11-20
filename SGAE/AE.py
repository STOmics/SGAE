from torch import nn
from torch.nn import Linear


class AE_encoder(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, n_inputs, n_z):
        super(AE_encoder, self).__init__()
        self.enc_1 = Linear(n_inputs, ae_n_enc_1)
        self.enc_2 = Linear(ae_n_enc_1, ae_n_enc_2)
        self.enc_3 = Linear(ae_n_enc_2, ae_n_enc_3)
        self.z_layer = Linear(ae_n_enc_3, n_z)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    # def forward(self, x):
    #     z = self.act(self.enc_1(x))
    #     z = self.act(self.enc_2(z))
    #     z = self.act(self.enc_3(z))
    #     z = self.z_layer(z)
    #     return z
    def forward(self, x):
        x = self.act(self.enc_1(x))
        x = self.act(self.enc_2(x))
        x = self.act(self.enc_3(x))
        x = self.z_layer(x)
        return x

class AE_decoder(nn.Module):

    def __init__(self, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_inputs, n_z):
        super(AE_decoder, self).__init__()
        self.dec_1 = Linear(n_z, ae_n_dec_1)
        self.dec_2 = Linear(ae_n_dec_1, ae_n_dec_2)
        self.dec_3 = Linear(ae_n_dec_2, ae_n_dec_3)
        self.x_bar_layer = Linear(ae_n_dec_3, n_inputs)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, z_ae):
        #     z = self.act(self.dec_1(z_ae))
        #     z = self.act(self.dec_2(z))
        #     z = self.act(self.dec_3(z))
        #     z = self.x_bar_layer(z)
        #     return z
        z_ae = self.act(self.dec_1(z_ae))
        z_ae = self.act(self.dec_2(z_ae))
        z_ae = self.act(self.dec_3(z_ae))
        z_ae = self.x_bar_layer(z_ae)
        return z_ae

class AE(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_inputs, n_z):
        super(AE, self).__init__()

        self.encoder = AE_encoder(ae_n_enc_1=ae_n_enc_1, ae_n_enc_2=ae_n_enc_2, ae_n_enc_3=ae_n_enc_3,
                                  n_inputs=n_inputs, n_z=n_z)

        self.decoder = AE_decoder(ae_n_dec_1=ae_n_dec_1, ae_n_dec_2=ae_n_dec_2, ae_n_dec_3=ae_n_dec_3,
                                  n_inputs=n_inputs, n_z=n_z)

    # def forward(self, x):
    #     z_ae = self.encoder(x)
    #     x_hat = self.decoder(z_ae)
    #     return x_hat, z_ae
    def forward(self, x):
        x = self.encoder(x)
        x_hat = self.decoder(x)
        return x_hat, x

# class AE_tol(nn.Module):
#     # 中间层亦是预测层
#     def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_inputs, n_z):
#         super(AE_tol, self).__init__()
#
#         self.encoder = AE_encoder(ae_n_enc_1=ae_n_enc_1, ae_n_enc_2=ae_n_enc_2, ae_n_enc_3=ae_n_enc_3,
#                                   n_inputs=n_inputs, n_z=n_z)
#
#         self.decoder = AE_decoder(ae_n_dec_1=ae_n_dec_1, ae_n_dec_2=ae_n_dec_2, ae_n_dec_3=ae_n_dec_3,
#                                   n_inputs=n_inputs, n_z=n_z)
#
#     def forward(self, x):
#         z_ae = self.encoder(x)
#         x_hat = self.decoder(z_ae)
#         return x_hat, self.s(z_ae)
