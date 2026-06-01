"""
BIST 1D-CNN Double DQN — Eğitim Grafik Analiz Modülü
Yazar: Fettah & AI Pair Programmer

Bu modül; terminaldeki eğitim çıktılarını (logs) okuyarak Portföy Değeri,
Loss, Epsilon ve Sharpe oranı değişimlerini şık grafiklerle görselleştirir.
"""

import os
import re
import matplotlib.pyplot as plt


def main() -> None:
    log_file = os.path.join("outputs", "train.log")
    if not os.path.exists(log_file):
        print(f"Hata: '{log_file}' dosyası bulunamadı!")
        print("-" * 75)
        print("Nasıl Kullanılır?")
        print("1. Terminalindeki tüm eğitim çıktılarını (Ctrl+A -> Ctrl+C ile) kopyala.")
        print("2. Proje dizinindeki 'outputs/' klasörü altında 'train.log' isimli bir dosya aç.")
        print("3. Kopyaladığın logları bu dosyanın içine yapıştırıp kaydet.")
        print("4. Ardından bu script'i çalıştır: python src/models/cnn/plot_logs.py")
        print("-" * 75)
        return

    episodes: list[int] = []
    portfolios: list[float] = []
    returns: list[float] = []
    sharpes: list[float] = []
    mdds: list[float] = []
    losses: list[float] = []
    epsilons: list[float] = []

    # Regex: Log satırlarını yakalamak için tasarlanmıştır:
    # Episode  45/100 | Port:  14,250.30 TL | Getiri:  +42.50% | Sharpe: +1.120 | MDD:  12.40% | İşlem:  45 | Loss: 0.1245 | ε: 0.7978
    pattern = re.compile(
        r"Episode\s+(\d+)/\d+\s+\|\s+Port:\s+([\d,.]+)\s+TL\s+\|\s+Getiri:\s+([+-]?[\d.]+)\%\s+\|\s+Sharpe:\s+([+-]?[\d.]+)\s+\|\s+MDD:\s+([+-]?[\d.]+)\%\s+\|\s+.*Loss:\s+([\d.]+)\s+\|\s+ε:\s+([\d.]+)"
    )

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                episodes.append(int(match.group(1)))
                portfolios.append(float(match.group(2).replace(",", "")))
                returns.append(float(match.group(3)))
                sharpes.append(float(match.group(4)))
                mdds.append(float(match.group(5)))
                losses.append(float(match.group(6)))
                epsilons.append(float(match.group(7)))

    if not episodes:
        print("Hata: 'outputs/train.log' dosyası içinde uygun eğitim satırları bulunamadı!")
        print("Lütfen logların doğru formatta yapıştırıldığından emin ol.")
        return

    print(f"✓ Toplam {len(episodes)} bölümlük log verisi başarıyla okundu.")
    print("Grafik çiziliyor...")

    # Görselleştirme Tasarımı (Sleek Modern Theme)
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Renk Paletimiz
    c_port = "#10b981"  # Emerald Green (Kâr rengi)
    c_loss = "#f43f5e"  # Rose Red (Hata rengi)
    c_eps = "#3b82f6"   # Blue (Keşif rengi)
    c_sharpe = "#a855f7"  # Purple (Sharpe rengi)

    # 1. Grafik: Portföy Değeri Gelişimi
    axs[0].plot(episodes, portfolios, color=c_port, linewidth=2, label="Portföy Değeri (TL)")
    axs[0].axhline(y=10000.0, color="#6b7280", linestyle="--", alpha=0.8, label="Başlangıç Sermayesi (10K)")
    axs[0].set_title("1D-CNN Ajanı Portföy Değeri Gelişimi", fontsize=13, fontweight="bold", pad=8)
    axs[0].set_ylabel("Portföy Değeri (TL)", fontsize=11)
    axs[0].legend(loc="upper left")
    axs[0].grid(True, linestyle=":", alpha=0.6)

    # 2. Grafik: Loss (Kayıp) Eğrisi
    axs[1].plot(episodes, losses, color=c_loss, linewidth=2, label="Eğitim Kaybı (Loss)")
    axs[1].set_title("Sinir Ağı Hata Payı (Loss) Eğrisi", fontsize=13, fontweight="bold", pad=8)
    axs[1].set_ylabel("Loss Seviyesi", fontsize=11)
    axs[1].legend(loc="upper right")
    axs[1].grid(True, linestyle=":", alpha=0.6)

    # 3. Grafik: Epsilon & Sharpe Oranı
    axs[2].plot(episodes, epsilons, color=c_eps, linewidth=2, label="Epsilon (Keşif Oranı)")
    ax2_twin = axs[2].twinx()
    ax2_twin.plot(episodes, sharpes, color=c_sharpe, linewidth=1.5, linestyle=":", label="Sharpe Oranı")
    
    axs[2].set_title("Keşif Oranı (Epsilon) ve Sharpe Rasyosu Karşılaştırması", fontsize=13, fontweight="bold", pad=8)
    axs[2].set_xlabel("Episode (Bölüm)", fontsize=11)
    axs[2].set_ylabel("Epsilon (Rastgelelik)", color=c_eps, fontsize=11)
    ax2_twin.set_ylabel("Sharpe Oranı", color=c_sharpe, fontsize=11)

    # İki farklı eksendeki legend'ları birleştirme
    lines1, labels1 = axs[2].get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    axs[2].legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    axs[2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_png = os.path.join("outputs", "training_curves.png")
    plt.savefig(output_png, dpi=300)
    plt.close()

    print(f"✓ Muhteşem! Grafik çizildi ve kaydedildi → {output_png}")
    print("Görseli incelemek için 'outputs/training_curves.png' dosyasını açabilirsin!")


if __name__ == "__main__":
    main()
