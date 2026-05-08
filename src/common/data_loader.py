"""
Borsa İstanbul (BIST) Reinforcement Learning Veri Yükleyici Modülü
Yazar: Senior AI Engineer

Bu modül; MLP, CNN, LSTM, GRU ve Dueling DQN modellerinin tümüne standartlaştırılmış
3D Tensör boyutunda veri sağlamakla yükümlüdür. Finansal mühendislik ve Anti-Data Leakage
prensipleri gözetilerek tasarlanmıştır.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler


class BISTDataLoader:
    """
    BIST hisse senedi verilerini modeller (LSTM, GRU, 1D-CNN, DQN, MLP) için 
    hazırlayan merkezi veri ardışık düzeni (pipeline) sınıfı.
    """

    def __init__(self, data_dir: str = "data", window_size: int = 30, test_split: float = 0.2):
        """
        BISTDataLoader sınıfını başlatır.

        Args:
            data_dir (str): CSV dosyalarının bulunduğu klasör yolu.
            window_size (int): Kayan pencere boyutu (zamansal adım sayısı).
            test_split (float): Test setine ayrılacak veri oranı (0.0 - 1.0 arası).
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.test_split = test_split
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_data(self, symbol: str) -> pd.DataFrame:
        """
        Ham CSV dosyasını okur, kronolojik olarak sıralar ve eksik/hatalı verileri onarır.
        
        - Hacmi (Volume) 0 olan, piyasanın kapalı olduğu günleri veri setinden çıkarır.
        - Look-ahead bias'ı önlemek için NaN değerleri SADECE FFill (İleri Doldurma) ile temizler.
        - FFill sonrası en başta kalan (geçmişi olmayan) NaN satırlarını düşer (dropna).

        Args:
            symbol (str): İşlenecek hisse senedi sembolü (Örn: 'THYAO').

        Returns:
            pd.DataFrame: Temizlenmiş ve sıralanmış pandas DataFrame.
            
        Raises:
            FileNotFoundError: Eğer belirtilen sembole ait veri dosyası bulunamazsa.
        """
        file_path = os.path.join(self.data_dir, f"{symbol}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Veri dosyası bulunamadı: '{file_path}'")

        # Veriyi yükle
        df = pd.read_csv(file_path)

        # Tarihe göre kronolojik sıralama (Süreklilik ve nedensellik için kritik)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values('Date', ascending=True, inplace=True)
            df.set_index('Date', inplace=True)

        # 1. Hacim (Volume) 0 olan satırların çıkarılması (Piyasanın kapalı olduğu sentetik günler)
        if 'Volume' in df.columns:
            df = df[df['Volume'] > 0]

        # 2. Look-ahead Bias (Gelecek sızıntısı) engelleme: Sadece Forward Fill kullan
        df = df.ffill()

        # 3. FFill yapılmasına rağmen ilk satırlarda kalan geçmişsiz verilerin silinmesi
        df.dropna(inplace=True)

        return df

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        pandas_ta kullanarak durum uzayına (state space) teknik göstergeleri ekler.
        
        Eklenen Göstergeler:
        - SMA_20 ve SMA_50 (Trend)
        - RSI_14 (Momentum)
        - MACD_12_26_9 (Momentum)

        Args:
            df (pd.DataFrame): Temel OHLCV verisini içeren DataFrame.

        Returns:
            pd.DataFrame: Teknik göstergelerin eklendiği genişletilmiş DataFrame.
        """
        # Kopya üzerinde çalışarak orijinal ham veriyi koruyoruz
        data = df.copy()

        # Trend İndikatörleri: SMA (Simple Moving Average)
        data['SMA_20'] = ta.sma(data['Close'], length=20)
        data['SMA_50'] = ta.sma(data['Close'], length=50)

        # Momentum İndikatörü: RSI (Relative Strength Index)
        data['RSI_14'] = ta.rsi(data['Close'], length=14)
        
        # MACD İndikatörü
        # ta.macd varsayılan olarak MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9 döner.
        macd_df = ta.macd(data['Close'], fast=12, slow=26, signal=9)
        if macd_df is not None:
            data = pd.concat([data, macd_df], axis=1)

        # Gösterge hesaplamalarından kaynaklanan ilk günlerdeki NaN değerlerini düşüyoruz
        # (Örn: SMA_50 için ilk 49 gün NaN olacaktır)
        data.dropna(inplace=True)
        return data

    def scale_and_window(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Anti-Data Leakage kuralına uygun ölçeklendirme (MinMaxScaler) yapar ve 
        modeller için (Samples, Window_Size, Features) boyutunda 3D tensörler oluşturur.

        İşlem Adımları:
        1. Train/Test ayrım noktası belirlenir (shuffle=False kuralı).
        2. Scaler SADECE Train setinde fit edilir. Gelecek verisi eğitime sızmaz.
        3. Tüm veri bu parametrelerle transform edilir.
        4. Belirlenen pencere boyutunda kayan pencereler oluşturulur.
        5. Pencereler Train ve Test olarak bölünür.

        Args:
            df (pd.DataFrame): Göstergelerin eklendiği, temizlenmiş veri seti.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (X_train, X_test) 3D numpy tensörleri.
        """
        # String vb. sızmış olabilecek sütunlara karşı güvenlik önlemi (Sadece sayısalları al)
        numeric_df = df.select_dtypes(include=[np.number])
        data_values = numeric_df.values

        # Zaman serisi kronolojisine uygun (shuffle=False) Split Indeksi
        split_idx = int(len(data_values) * (1 - self.test_split))

        # Kritik Mühendislik Kuralı: Scaler'ı SADECE Train verisi üzerinde fit et!
        train_data = data_values[:split_idx]
        self.scaler.fit(train_data)

        # Fit edilen scaler ile TÜM veriyi transform et
        scaled_data = self.scaler.transform(data_values)

        # 3D Tensor Yapılandırması (Windowing)
        # Hedef Boyut: (Num_Samples, Window_Size, Num_Features)
        windows = []
        for i in range(len(scaled_data) - self.window_size + 1):
            window = scaled_data[i : i + self.window_size]
            windows.append(window)

        windows = np.array(windows)

        # Oluşan pencereleri zaman sıralamasını bozmadan (shuffle=False) ikiye böl
        window_split_idx = int(len(windows) * (1 - self.test_split))

        X_train = windows[:window_split_idx]
        X_test = windows[window_split_idx:]

        return X_train, X_test

    def get_pipeline_data(self, symbol: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ham CSV dosyasından modeller için hazır 3D tensöre uzanan tam süreci (pipeline) tek kalemde yönetir.

        Args:
            symbol (str): İşlenecek hisse senedi sembolü (Örn: 'THYAO').

        Returns:
            Tuple[np.ndarray, np.ndarray]: X_train ve X_test 3D Numpy tensörleri.
        """
        # 1. Dosya Yolu ve Veri Yükleme (Eksik veriler temizlenir)
        df_raw = self.load_data(symbol)
        
        # 2. Öznitelik Mühendisliği (Teknik indikatörler eklenir)
        df_features = self.add_indicators(df_raw)
        
        # 3. Sızıntı Önleyici Normalizasyon ve 3D Tensor Dönüşümü
        X_train, X_test = self.scale_and_window(df_features)
        
        return X_train, X_test
