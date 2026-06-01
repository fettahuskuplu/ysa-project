import torch
import torch.nn as nn

class MLPQNetwork(nn.Module):
    """
    MLP-DQN için Çok Katmanlı Algılayıcı Mimari Yapısı.
    """
    def __init__(self, window_size: int, num_features: int, action_size: int = 3):
        super(MLPQNetwork, self).__init__()
        
        # data_loader'dan gelen 3D yapıyı (Window_Size * Features) şeklinde düzleştireceğiz
        self.input_dim = window_size * num_features
        
        self.network = nn.Sequential(
            nn.Flatten(), # (Batch, Window_Size, Features) -> (Batch, Window_Size * Features) yapar.
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2), # Finansal veride ezberlemeyi (overfitting) engellemek için kritik
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size) # Çıktı: HOLD, BUY, SELL aksiyonlarının Q-değerleri
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Girdi Numpy dizisi olarak gelirse tensöre güvenli dönüşüm garantisi
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
            
        # Eğer tek bir anlık durum (state) gelirse batch boyutunu 1 yapma güvenliği (2D -> 3D)
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
            
        return self.network(state)