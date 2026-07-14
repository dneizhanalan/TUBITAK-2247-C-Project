# gecikme_taramasi.py
# Sinaptik gecikmeyi (tau_d) yavas yavas artiriyorum ve iki seyi izliyorum:
#  1) hangi gecikmede sistem salinima basliyor (Hopf catallanmasi)
#  2) salinim basladiktan sonra frekans gecikmeyle nasil degisiyor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import get_params, baslangic_durumu
from rk4_cozucu import simule_et
from spektral_analiz import gama_analizi


def genlik_tarama(tau_d_listesi, t_max=3000.0, dt=0.05, gecici=2000.0):
    genlikler = []
    for tau_d in tau_d_listesi:
        p = get_params()
        p["tau_d"] = tau_d
        x0 = baslangic_durumu(p)
        t, X = simule_et(p, x0, t_max, dt)
        r = X[:, 0]
        r_son = r[t > gecici]
        genlik = r_son.max() - r_son.min()
        genlikler.append(genlik)
    return np.array(genlikler)


def hopf_esigini_bul(tau_d_listesi, genlikler, kat=3.0):
    # baslangictaki (kucuk gecikmelerdeki) ortalama genligin "kat" kati
    # asildiginda Hopf basladi diyoruz
    taban = np.mean(genlikler[:3])
    esik = kat * taban

    for i in range(len(genlikler)):
        if genlikler[i] > esik:
            if i == 0:
                return tau_d_listesi[0]
            # iki nokta arasinda basit interpolasyon
            x0, x1 = tau_d_listesi[i - 1], tau_d_listesi[i]
            y0, y1 = genlikler[i - 1], genlikler[i]
            oran = (esik - y0) / (y1 - y0)
            return x0 + oran * (x1 - x0)
    return None


if __name__ == "__main__":
    # 1. adim: kaba tarama ile Hopf esigini bul
    tau_kaba = np.arange(0.0, 10.01, 0.25)
    genlikler = genlik_tarama(tau_kaba)
    tau_hopf = hopf_esigini_bul(tau_kaba, genlikler)
    print(f"Hopf esigi yaklasik: tau_d = {tau_hopf:.2f} ms")

    # 2. adim: esik ustunde frekans nasil degisiyor bakalim
    tau_ince = np.arange(np.ceil(tau_hopf) + 0.5, 20.01, 1.0)
    frekanslar = []
    for tau_d in tau_ince:
        p = get_params()
        p["tau_d"] = tau_d
        x0 = baslangic_durumu(p)
        dt = 0.02
        t, X = simule_et(p, x0, t_max=3000.0, dt=dt)
        r = X[:, 0]
        freq, psd, tepe = gama_analizi(t, r, dt, gecici_sure=2000.0)
        frekanslar.append(tepe)

    print("\ntau_d (ms) -> tepe frekans (Hz)")
    for td, f in zip(tau_ince, frekanslar):
        print(f"  {td:.2f} -> {f:.2f}")

    fig, eksen = plt.subplots(1, 2, figsize=(11, 4.5))

    eksen[0].plot(tau_kaba, genlikler, "o-")
    eksen[0].axvline(tau_hopf, color="red", linestyle="--", label=f"Hopf ~{tau_hopf:.2f} ms")
    eksen[0].set_yscale("log")
    eksen[0].set_xlabel("tau_d (ms)")
    eksen[0].set_ylabel("genlik")
    eksen[0].set_title("Gecikmeye gore osilasyon genligi")
    eksen[0].legend()

    eksen[1].plot(tau_ince, frekanslar, "s-", color="darkred")
    eksen[1].axhspan(30, 100, alpha=0.12, label="gama bant")
    eksen[1].set_xlabel("tau_d (ms)")
    eksen[1].set_ylabel("frekans (Hz)")
    eksen[1].set_title("Gecikmeye gore gama frekansi")
    eksen[1].legend()

    plt.tight_layout()

    import os
    klasor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
    os.makedirs(klasor, exist_ok=True)
    plt.savefig(os.path.join(klasor, "gecikme_taramasi.png"), dpi=150)
    print("\ngrafik kaydedildi: figs/gecikme_taramasi.png")
