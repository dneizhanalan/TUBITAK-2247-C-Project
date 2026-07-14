# spektral_analiz.py
# r(t) sinyalinin frekans icerigine bakiyorum (Welch yontemiyle PSD/guc
# spektrumu hesapliyorum). Amac gama bandinda (30-100 Hz) bir tepe olup
# olmadigini gormek.
#
# fs (ornekleme frekansi) hesabi: dt ms cinsinden oldugu icin
# fs = 1000/dt yapiyorum (saniyede kac ornek aldigimizi buluyoruz).

import numpy as np
from scipy.signal import welch


def gama_analizi(t, r, dt, gecici_sure=200.0):
    fs = 1000.0 / dt

    # ilk kisim gecici rejim (transient), onu atip son kismi kullaniyoruz
    r_son = r[t >= gecici_sure]

    n = min(4096, len(r_son))
    freq, psd = welch(r_son - np.mean(r_son), fs=fs, nperseg=n)

    # 30-100 Hz arasindaki en buyuk tepeyi buluyoruz
    bant = (freq >= 30) & (freq <= 100)
    if np.any(bant):
        idx = np.argmax(psd[bant])
        tepe_freq = freq[bant][idx]
    else:
        tepe_freq = np.nan

    return freq, psd, tepe_freq


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from model import get_params, baslangic_durumu
    from rk4_cozucu import simule_et

    p = get_params()
    p["tau_d"] = 8.0   # gecikme ekleyince salinim basliyor (bkz. gecikme_taramasi.py)

    x0 = baslangic_durumu(p)
    dt = 0.02
    t, X = simule_et(p, x0, t_max=2500.0, dt=dt)
    r = X[:, 0]

    freq, psd, tepe_freq = gama_analizi(t, r, dt)
    print(f"gama bandindaki tepe frekans: {tepe_freq:.2f} Hz")

    fig, eksen = plt.subplots(1, 2, figsize=(10, 4))

    eksen[0].plot(t, r, linewidth=0.7)
    eksen[0].set_xlabel("zaman (ms)")
    eksen[0].set_ylabel("r(t)")
    eksen[0].set_title(f"tau_d = {p['tau_d']} ms")

    eksen[1].semilogy(freq, psd)
    eksen[1].axvspan(30, 100, alpha=0.15, label="gama bant")
    eksen[1].axvline(tepe_freq, linestyle="--", color="black")
    eksen[1].set_xlim(0, 150)
    eksen[1].set_xlabel("frekans (Hz)")
    eksen[1].set_ylabel("guc")
    eksen[1].legend()

    plt.tight_layout()

    import os
    klasor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
    os.makedirs(klasor, exist_ok=True)
    plt.savefig(os.path.join(klasor, "spektral_analiz.png"), dpi=150)
    print("grafik kaydedildi: figs/spektral_analiz.png")
