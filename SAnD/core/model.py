import torch
import torch.nn as nn
from ..core import modules
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayerForSAnDImprove(nn.Module):
    def __init__(self, input_features, seq_len, n_heads, n_layers, d_model=256, dropout_rate=0.1) -> None:
        super(EncoderLayerForSAnDImprove, self).__init__()
        self.d_model = d_model

        self.input_embedding = nn.Conv1d(input_features, d_model, kernel_size=3, padding=1)
        self.positional_encoding = modules.PositionalEncoding(d_model, seq_len)

        # Aumentar el número de bloques
        self.blocks = nn.ModuleList([modules.EncoderBlock(d_model, n_heads, dropout_rate) for _ in range(n_layers)])

        # Capa densa intermedia (entre los bloques)
        self.intermediate_dense = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(d_model * 2),  # Añadir Batch Normalization
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_embedding(x)
        x = x.transpose(1, 2)

        x = self.positional_encoding(x)

        for l in self.blocks:
            x = l(x)

        # Aplicar la capa intermedia correctamente
        x = self.intermediate_dense[0](x)  # Linear(d_model, d_model * 2)
        x = self.intermediate_dense[1](x)  # ReLU

        x = x.permute(0, 2, 1)  # (batch_size, d_model * 2, seq_len)
        x = self.intermediate_dense[2](x)  # BatchNorm1d(d_model * 2)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, d_model * 2)

        x = self.intermediate_dense[3](x)  # Dropout
        x = self.intermediate_dense[4](x)  # Linear(d_model * 2, d_model)

        return x

class SAnDImprove(nn.Module):
    """
    Simply Attend and Diagnose model

    The Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18)

    `Attend and Diagnose: Clinical Time Series Analysis Using Attention Models <https://arxiv.org/abs/1711.03905>`_
    Huan Song, Deepta Rajan, Jayaraman J. Thiagarajan, Andreas Spanias
    """

    def __init__(
            self, input_features: int, seq_len: int, n_heads: int, factor: int,
            n_class: int, n_layers: int, d_model: int = 128, dropout_rate: float = 0.1

    ) -> None:
        super(SAnDImprove, self).__init__()
        self.encoder = EncoderLayerForSAnDImprove(input_features, seq_len, n_heads, n_layers, d_model, dropout_rate)
        self.dense_interpolation = modules.DenseInterpolation(seq_len, factor)
        self.clf = modules.ClassificationModule(d_model, factor, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.dense_interpolation(x)
        x = self.clf(x)
        return torch.sigmoid(x)  # Aplicamos sigmoide en la salida final





class EncoderLayerForSAnD(nn.Module):
    def __init__(self, input_features, seq_len, n_heads, n_layers, d_model=128, dropout_rate=0.2) -> None:
        super(EncoderLayerForSAnD, self).__init__()
        self.d_model = d_model

        self.input_embedding = nn.Conv1d(input_features, d_model, 1)
        self.positional_encoding = modules.PositionalEncoding(d_model, seq_len)
        self.blocks = nn.ModuleList([
            modules.EncoderBlock(d_model, n_heads, dropout_rate) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = x.unsqueeze(0)  # Ahora x1 tiene la forma (1, 400, 3)
        x = x.transpose(1, 2)
        x = self.input_embedding(x)
        x = x.transpose(1, 2)

        x = self.positional_encoding(x)

        for l in self.blocks:
            x = l(x)

        return x


class SAnD(nn.Module):
    """
    Simply Attend and Diagnose model

    The Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18)

    `Attend and Diagnose: Clinical Time Series Analysis Using Attention Models <https://arxiv.org/abs/1711.03905>`_
    Huan Song, Deepta Rajan, Jayaraman J. Thiagarajan, Andreas Spanias
    """

    def __init__(
            self, input_features: int, seq_len: int, n_heads: int, factor: int,
            n_class: int, n_layers: int, d_model: int = 128, dropout_rate: float = 0.2

    ) -> None:
        super(SAnD, self).__init__()
        self.encoder = EncoderLayerForSAnD(input_features, seq_len, n_heads, n_layers, d_model, dropout_rate)
        self.dense_interpolation = modules.DenseInterpolation(seq_len, factor)
        self.clf = modules.ClassificationModule(d_model, factor, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.dense_interpolation(x)
        x = self.clf(x)
        return x


class SAnD_Embedding(nn.Module):
    """
    Versión mejorada del modelo siames con embeddings optimizados
    """
    def __init__(self, input_features: int, seq_len: int, n_heads: int, factor: int,
                 n_class: int, n_layers: int, d_model: int = 128, dropout_rate: float = 0.2) -> None:
        super(SAnD_Embedding, self).__init__()
        self.encoder = EncoderLayerForSAnD(input_features, seq_len, n_heads, n_layers, d_model, dropout_rate)
        self.dense_interpolation = modules.DenseInterpolation(seq_len, factor)

        # Embedding mejorado con una capa oculta adicional
        self.embedding_layer = nn.Sequential(
            nn.Linear(d_model * factor, 256),  # Aumentamos dimensionalidad
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(256),
            nn.Linear(256, 1)  # Reducimos de nuevo a 128
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.dense_interpolation(x)
        x = x.reshape(x.size(0), -1)
        x = self.embedding_layer(x)  # Embedding final mejorado
        return x

class SiameseSAnD(nn.Module):
    """
    Modelo siamesa con comparación basada en similitud coseno
    """
    def __init__(self, sand_model: SAnD_Embedding):
        super(SiameseSAnD, self).__init__()
        self.sand = sand_model

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        emb1 = self.sand(x1)
        emb2 = self.sand(x2)
        similarity = F.cosine_similarity(emb1, emb2)  # Similitud coseno en vez de distancia euclidiana
        return similarity


###########################################################################################
##################   NARX para 2 variables (V, I)
##########################################################################################

class NARX_Transformer_2var(nn.Module):
    def __init__(self, feature_dim1,feature_dim2, num_attention, num_cycles, num_preds):
        super(NARX_Transformer_2var, self).__init__()
        self.num_cycles = num_cycles
        self.num_preds = num_preds
        self.cap_linear_layer = nn.Linear(self.num_cycles-1, feature_dim2)
        self.final_linear_layer = nn.Linear(feature_dim2, 1)

        # self.conv_layer = nn.Conv1d(3, 512, kernel_size=16, stride=8)
        self.conv_layer = nn.Conv2d(in_channels=2, out_channels=feature_dim1, kernel_size=3, stride=1, padding=1)
        self.conv_layer2 = nn.Conv2d(in_channels=feature_dim1, out_channels=feature_dim2, kernel_size=3, padding=1)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim2, nhead=num_attention, batch_first=True)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim2, nhead=num_attention, batch_first=True)

    def forward(self, my_data, capacity):
        my_data = my_data.permute(0, 3, 2, 1)  # => (batch, channels=2, height=400, width=2)
        embedded_data = self.conv_layer(my_data)
        embedded_data = self.conv_layer2(embedded_data)

        # Aplanamos las dimensiones espaciales (height, width) en una sola dimensión
        batch_size, channels, h, w = embedded_data.shape
        embedded_data = embedded_data.view(batch_size, channels, h * w)  # (B, feature_dim2, H'*W')

        # Ahora permutamos para que sea (B, seq_len, feature_dim2)
        embedded_data = embedded_data.permute(0, 2, 1)

        encoded_data = self.encoder_layer(embedded_data)

        tgt = self.cap_linear_layer(capacity)
        tgt = tgt.unsqueeze(1)
        decoded_data = self.decoder_layer(tgt, encoded_data)
        decoded_data = decoded_data.squeeze(1)

        output_cap = self.final_linear_layer(decoded_data)

        return output_cap

    def pred_sequence(self, my_data, capacity):
        pred_caps = torch.stack([capacity[:,i] for i in range(self.num_cycles-1)], axis=-1)
        for cycle in range(self.num_preds):
            pred = self.forward(my_data[:,cycle:cycle+self.num_cycles], pred_caps[:,-self.num_cycles+1:])
            pred_caps = torch.cat([pred_caps, pred], axis=-1)
        return pred_caps
# #


class NARX_Transformer_2var_SoloActual(nn.Module):
    def __init__(self, feature_dim1, feature_dim2, num_attention, num_cycles, num_preds):
        super(NARX_Transformer_2var_SoloActual, self).__init__()
        self.num_cycles = num_cycles
        self.num_preds = num_preds

        # Encoder CNN
        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=feature_dim1, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=feature_dim1, out_channels=feature_dim2, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim2,
            nhead=num_attention,
            batch_first=True,
            dim_feedforward=feature_dim2 * 4,
            dropout=0.1
        )

        # Head final para predicción
        self.fc = nn.Sequential(
            nn.Linear(feature_dim2, feature_dim2 // 2),
            nn.ReLU(),
            nn.Linear(feature_dim2 // 2, 1)
        )

    def forward(self, my_data):
        # my_data tiene forma (batch, 400, 2)
        current_cycle = my_data.permute(0, 2, 1)  # (batch, 2, 400)

        embedded_data = self.conv_layer(current_cycle)  # (batch, feature_dim2, 400)

        embedded_data = embedded_data.permute(0, 2, 1)  # (batch, 400, feature_dim2)

        encoded_data = self.encoder_layer(embedded_data)  # (batch, 400, feature_dim2)

        pooled = torch.mean(encoded_data, dim=1)  # (batch, feature_dim2)

        output_cap = self.fc(pooled)  # (batch, 1)

        return output_cap

    def pred_sequence(self, my_data):
        """
        Inferencia secuencial.
        Asume que my_data tiene forma (batch, total_ciclos, 400, 2)
        """
        preds = []

        total_ciclos = my_data.size(1)

        for cycle in range(total_ciclos):
            current_cycle = my_data[:, cycle, :, :]  # (batch, 400, 2)

            pred = self.forward(current_cycle)  # (batch, 1)
            preds.append(pred)

        preds = torch.cat(preds, dim=1)  # (batch, total_ciclos)

        return preds



#
#
#
# ###########################################################################################
# ##################   NARX para 3 variables (V, I, Tª)
# ##########################################################################################
class NARX_Transformer(nn.Module):
    def __init__(self, feature_dim1,feature_dim2, num_attention, num_cycles, num_preds):
        super(NARX_Transformer, self).__init__()
        self.num_cycles = num_cycles
        self.num_preds = num_preds
        self.cap_linear_layer = nn.Linear(self.num_cycles-1, feature_dim2)
        self.final_linear_layer = nn.Linear(feature_dim2, 1)

        # self.conv_layer = nn.Conv1d(3, 512, kernel_size=16, stride=8)
        self.conv_layer = nn.Conv2d(num_cycles, feature_dim1, kernel_size=3, stride=1,padding=1)
        self.conv_layer2 = nn.Conv2d(feature_dim1,feature_dim2,kernel_size=3)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim2, nhead=num_attention, batch_first=True)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim2, nhead=num_attention, batch_first=True)

    def forward(self, my_data, capacity):
        embedded_data = self.conv_layer(my_data)
        embedded_data = self.conv_layer2(embedded_data).squeeze(-1)
        embedded_data = embedded_data.permute(0, 2, 1)
        encoded_data = self.encoder_layer(embedded_data)

        tgt = self.cap_linear_layer(capacity)
        tgt = tgt.unsqueeze(1)
        decoded_data = self.decoder_layer(tgt, encoded_data)
        decoded_data = decoded_data.squeeze(1)
        output_cap = self.final_linear_layer(decoded_data)
        return output_cap

    def pred_sequence(self, my_data, capacity):
        pred_caps = torch.stack([capacity[:,i] for i in range(self.num_cycles-1)], axis=-1)
        for cycle in range(self.num_preds):
            pred = self.forward(my_data[:,cycle:cycle+self.num_cycles], pred_caps[:,-self.num_cycles+1:])
            pred_caps = torch.cat([pred_caps, pred], axis=-1)
        return pred_caps
