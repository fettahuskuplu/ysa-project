"""
Borsa İstanbul (BIST) Reinforcement Learning Alım-Satım Simülasyon Ortamı
Yazar: Senior AI Engineer & RL Takımı

Bu modül; Dueling DQN ve diğer Reinforcement Learning ajanlarının geçmiş piyasa 
verileri üzerinde işlem (trade) yaparak portföy optimizasyonunu öğrenebileceği 
production-grade simülasyon ortamıdır. OpenAI Gym standartlarına uygun API sağlar.
"""

import numpy as np
from typing import Tuple, Dict, Any, List


class TradingEnvironment:
    """
    Finansal piyasalar için özel olarak tasarlanmış Reinforcement Learning çevre (environment) sınıfı.
    
    Ajanın amacı, belirli bir başlangıç bakiyesi ile (initial_balance) alım (BUY), satım (SELL) 
    ve bekleme (HOLD) aksiyonlarını kullanarak portföy değerini (portfolio_value) maksimize etmektir.
    Ortam içerisinde açığa satış (short selling) yasaktır ve ajanın mevcut bakiyesinin altına 
    düşmesine (negatif bakiye) izin verilmez.
    """

    # --- Aksiyon Uzayı (Action Space) ---
    # 0: HOLD -> Mevcut pozisyonu koru (Hisse varsa tut, yoksa nakitte bekle).
    # 1: BUY  -> Mevcut tüm nakit ile (işlem ücreti düşüldükten sonra) hisse senedi al.
    # 2: SELL -> Eldeki tüm hisseleri piyasa fiyatından satarak nakde dön.
    ACTION_HOLD = 0
    ACTION_BUY = 1
    ACTION_SELL = 2

    def __init__(self, 
                 market_data: np.ndarray, 
                 real_prices: np.ndarray, 
                 initial_balance: float = 10000.0, 
                 transaction_fee_percent: float = 0.001, 
                 window_size: int = 30):
        """
        Trading ortamını RL ajanının etkileşimine hazır hale getirir.

        Args:
            market_data (np.ndarray): (Samples, Window_Size, Features) boyutunda normalize edilmiş state matrisi.
            real_prices (np.ndarray): (Samples,) boyutunda gerçek kapanış fiyatları (Ödül ve portföy hesabı için).
            initial_balance (float): Ajanın simülasyona başlarken sahip olduğu nakit para.
            transaction_fee_percent (float): Her alım-satım işleminde kesilecek komisyon oranı (Örn: binde 1 -> 0.001).
            window_size (int): Ajanın geçmişe dönük olarak görebileceği zaman adımı (state boyutu).
        """
        # Veri setleri (Anti-Data Leakage prensibine uygun olarak hazırlanmış veriler)
        self.market_data = market_data
        self.real_prices = real_prices
        
        # Ortam Kuralları
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.window_size = window_size
        self.total_steps = len(self.market_data)

        # RL Durum (State) Değişkenleri
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.portfolio_value = self.initial_balance
        self.done = False

        # Analiz ve Çizim İçin Geçmiş (History)
        self.action_history: List[int] = []
        self.portfolio_history: List[float] = []
        self.reward_history: List[float] = []

    def reset(self) -> np.ndarray:
        """
        Ortamı ilk başlangıç durumuna döndürür. Yeni bir periyot (episode) başlatırken kullanılır.

        Returns:
            np.ndarray: Başlangıç state'i (window_size, num_features) formatında.
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.portfolio_value = self.initial_balance
        self.done = False

        # Geçmişi sıfırla ve başlangıç portföy değerini ekle
        self.action_history = []
        self.portfolio_history = [self.initial_balance]
        self.reward_history = []

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Ajanın o an görebileceği finansal teknik durumu getirir.

        Returns:
            np.ndarray: Normalize edilmiş mevcut gözlem (observation).
                        Boyut: (window_size, num_features)
        """
        # Veri setindeki mevcut adıma ait 2D matrisi (Window) döndür.
        return self.market_data[self.current_step]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Ajanın verdiği aksiyonu işler, portföyü günceller, işlem ücretlerini keser 
        ve finansal kurallara göre ödül (reward) hesabı yapar.

        Args:
            action (int): Ajanın seçtiği aksiyon (0: HOLD, 1: BUY, 2: SELL).

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: 
                - next_state (np.ndarray): Bir sonraki durum.
                - reward (float): Gerçekleştirilen aksiyonun kalitesini ölçen sayısal ödül.
                - done (bool): Simülasyonun (episode) bitip bitmediği.
                - info (Dict): Analiz için ekstra metrikleri barındıran sözlük.
        """
        # Adım kısıtlaması ve done bayrağı kontrolü
        if self.done:
            return self._get_state(), 0.0, self.done, {"msg": "Episode already finished."}

        # --- GERÇEK FİYAT KULLANIMI ---
        # Mevcut adımın işlem göreceği gerçek piyasa fiyatı (Real Close Price)
        # İşlemler normalize verilerle değil, asıl piyasa fiyatlarıyla yapılır.
        current_price = self.real_prices[self.current_step]
        
        # Portföy değerini işlemden önce kaydet (Büyüme/küçülme hesabı için)
        prev_portfolio_value = self.portfolio_value

        # --- AKSİYONLARIN İŞLENMESİ VE RİSK YÖNETİMİ ---
        # 1. Geçersiz aksiyon kontrolü
        if action not in [self.ACTION_HOLD, self.ACTION_BUY, self.ACTION_SELL]:
            action = self.ACTION_HOLD # Fallback güvenliği
            
        trade_penalty = 0.0

        if action == self.ACTION_BUY:
            # Sadece yeterli bakiye varsa alım yapılabilir (No negative balance)
            if self.balance > 0:
                # Toplam nakdin ne kadarıyla hisse alınabileceğini hesapla (Komisyon düşülür)
                affordable_amount = self.balance * (1 - self.transaction_fee_percent)
                shares_bought = affordable_amount / current_price
                
                # Cüzdanı güncelle
                self.shares_held += shares_bought
                self.balance -= (shares_bought * current_price) * (1 + self.transaction_fee_percent)
            else:
                # Para yokken almaya çalışmak, anlamsız bir işlem cezası (Overtrading / Invalid action)
                trade_penalty = -0.5 

        elif action == self.ACTION_SELL:
            # Sadece elde hisse varsa satım yapılabilir (No short selling)
            if self.shares_held > 0:
                # Toplam hisse değerini hesapla ve komisyonu düş
                sale_value = self.shares_held * current_price
                cash_received = sale_value * (1 - self.transaction_fee_percent)
                
                # Cüzdanı güncelle
                self.balance += cash_received
                self.shares_held = 0.0
            else:
                # Hisse yokken satmaya çalışmak (Invalid action)
                trade_penalty = -0.5

        elif action == self.ACTION_HOLD:
            # Nakitteyken beklemenin fırsat maliyetini simüle etmek istenirse veya gereksiz
            # yere uzun süre pozisyonsuz kalınması cezalandırılmak istenirse burası kullanılabilir.
            # (Unnecessary HOLD penalty) - Şu an için risk sensitivity açısından hafif bir ceza uyguluyoruz:
            # Eğer elde hisse yoksa çok minimal bir ceza vererek işlem yapmaya teşvik edebiliriz.
            pass

        # --- PORTFÖY DEĞERLEMESİ ---
        # Güncel toplam varlık = Nakit + (Hisse Adedi * Güncel Fiyat)
        self.portfolio_value = self.balance + (self.shares_held * current_price)

        # --- FİNANSAL MÜHENDİSLİK: ÖDÜL (REWARD) HESAPLAMA ---
        # Ödül 1: Saf portföy değişimi (Kâr = +, Zarar = -)
        portfolio_change = self.portfolio_value - prev_portfolio_value
        
        # Ödül 2: Yüzdelik getiri (Portfolio Growth) - Büyük sermayelerde istikrar sağlar
        pct_return = portfolio_change / prev_portfolio_value if prev_portfolio_value > 0 else 0
        
        # Toplam Ödül (Risk sensitivity ile ölçeklenebilir)
        # Sadece TL bazında kar yerine, portföydeki %'lik büyüme Sharpe ratio mantığına daha uygundur.
        # İlerleyen safhalarda maksimum düşüşe (drawdown) dayalı ekstra risk cezaları eklenebilir.
        reward = (pct_return * 100.0) + trade_penalty

        # Geçmişe (Logging) aksiyon, portföy değeri ve ödülü ekle
        self.action_history.append(action)
        self.portfolio_history.append(self.portfolio_value)
        self.reward_history.append(reward)

        # Zamanı bir adım ileri sar
        self.current_step += 1

        # Episode Bitiş Kontrolü
        # Tüm veri seti bittiyse veya ajan sermayesinin %90'ını kaybettiyse (Ruin/İflas durumu)
        if self.current_step >= self.total_steps - 1 or self.portfolio_value < self.initial_balance * 0.1:
            self.done = True
            
            # Episode sonunda eğer kalan hissesi varsa sentetik olarak satıp nakde çevir
            if self.shares_held > 0:
                final_price = self.real_prices[self.current_step]
                sale_value = self.shares_held * final_price
                self.balance += sale_value * (1 - self.transaction_fee_percent)
                self.shares_held = 0.0
                self.portfolio_value = self.balance
                self.portfolio_history.append(self.portfolio_value)

        # Sonraki durum (Next State)
        next_state = self._get_state() if not self.done else np.zeros_like(self.market_data[0])
        
        # Ajan analizi ve dashboard için Info Dictionary
        info = {
            "step": self.current_step,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "portfolio_value": self.portfolio_value,
            "total_profit": self.portfolio_value - self.initial_balance,
            "action_taken": action
        }

        return next_state, float(reward), self.done, info
