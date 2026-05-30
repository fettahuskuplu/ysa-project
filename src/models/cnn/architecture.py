"""
Borsa İstanbul (BIST) 1D-CNN-DQN Model Mimarisi
Yazar: Fettah & AI Pair Programmer

Bu modül; zaman serilerindeki yerel örüntüleri (trendler, momentum vb.) yakalamak amacıyla 
tasarlanmış 1D-CNN (Tek Boyutlu Evrişimli Sinir Ağı) mimarisini barındırır.
"""

import torch
import torch.nn as nn
from src.models.cnn.config import CNNDQNConfig


class CNNDQNNetwork(nn.Module):
    """
    BIST zaman serisi verileri üzerinden 1D Evrişimler yardımıyla öznitelik çıkarımı yapan 
    ve her trading kararı (HOLD, BUY, SELL) için beklenen Q-değerlerini üreten ağ mimarisi.
    """

    def __init__(self, num_features: int, window_size: int, action_size: int) -> None:
        """
        1D-CNN katmanlarını ve karar kafasını (FC Layers) oluşturur.

        Args:
            num_features (int): Girdi özniteliklerinin sayısı (Örn: Close, Open, SMA, RSI vb. -> ~11)
            window_size (int): Kayan pencere boyutu (Zaman adımı sayısı -> Örn: 30 gün)
            action_size (int): Çıktı aksiyon sayısı (HOLD=0, BUY=1, SELL=2)
        """
        super(CNNDQNNetwork, self).__init__()

        hidden_size: int = CNNDQNConfig.HIDDEN_SIZE

        # PyTorch Conv1d girdileri (batch_size, in_channels, seq_len) formatında bekler.
        # Bizim durumumuzda:
        #   in_channels = num_features (Giriş kanalları/öznitelikler)
        #   seq_len = window_size (Zamansal uzunluk)
        # Bu nedenle forward adımında veriyi transpoze edeceğiz.

        # --- 1. Evrişim Bloğu ---
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)

        # --- 2. Evrişim Bloğu ---
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)

        # --- 3. Evrişim Bloğu ---
        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.2)

        # --- Küresel Ortalama Havuzlama (Global Average Pooling) ---
        # Zamansal boyuttaki (window_size) gürültüleri süzer, parametre sayısını azaltır 
        # ve aşırı öğrenmeyi (overfitting) çok büyük ölçüde önler.
        # Çıktı boyutu: (batch_size, 128, 1) olur.
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)

        # --- Karar Alma Kafası (Decision Head) ---
        self.fc = nn.Sequential(
            nn.Linear(in_features=128, out_features=hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=hidden_size, out_features=action_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        İleri besleme adımını yürütür.

        Args:
            x (torch.Tensor): (batch_size, window_size, num_features) boyutlu durum tensörü.

        Returns:
            torch.Tensor: Her aksiyon için (batch_size, action_size) boyutunda Q-değerleri.
        """
        # Girdi: (batch_size, window_size, num_features)
        # Conv1D için istenen: (batch_size, num_features, window_size)
        x = x.transpose(1, 2)

        # Evrişim, aktivasyon ve dropout adımları
        x = self.dropout1(self.relu1(self.conv1(x)))
        x = self.dropout2(self.relu2(self.conv2(x)))
        x = self.dropout3(self.relu3(self.conv3(x)))

        # Pooling (Zamansal sıkıştırma)
        x = self.pool(x)        # (batch_size, 128, 1)
        x = x.squeeze(dim=-1)   # (batch_size, 128)

        # Karar katmanı ile Q-değerlerini hesapla
        q_values: torch.Tensor = self.fc(x)
        return q_values
