# dt_yakinsama_testi.py
# Burda farkli dt (zaman adimi) degerleriyle ayni simulasyonu calistirip
# sonuclarin birbirine ne kadar yakin ciktigina bakiyorum.
#
# RK4 yontemi "4. derece" oldugu icin, dt'yi yariya indirince hatanin
# yaklasik 16 kat (2^4) kuculmesi lazim. Bunu grafikte kontrol ediyoruz.
#
# ONEMLI NOT: Ilk denedigimde hep buyuk bir dt (2 ms) kullanip cok uzun
# bir sure (200 ms) simule etmistim, ama sistem zaten dinlenme durumuna
# oturdugu icin hata cok kucuk kaliyordu ve duzgun bir "4. derece" trendi
# goremedim. Bunun yerine daha kisa bir sure (30 ms) kullandim, boylece
# sistem hala hareket halindeyken karsilastirma yapmis oldum. Bu sekilde
# beklenen egim (~4) net bir sekilde cikti.

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import get_params, baslangic_durumu
from rk4_cozucu import simule_et


def yakinsama_testi(p, t_max=30.0, dt_baslangic=2.0, kac_kere_yariya=7):
    x0 = baslangic_durumu(p)

    dt_listesi = []
    dt = dt_baslangic
    for i in range(kac_kere_yariya + 1):
        dt_listesi.append(dt)
        dt = dt / 2.0

    son_degerler = []
    for dt in dt_listesi:
        t, X = simule_et(p, x0, t_max, dt)
        son_degerler.append(X[-1])

    hatalar = []
    for i in range(len(dt_listesi) - 1):
        fark = np.linalg.norm(son_degerler[i] - son_degerler[i + 1])
        hatalar.append(fark)

    return dt_listesi[:-1], hatalar


if __name__ == "__main__":
    p = get_params()
    dt_listesi, hatalar = yakinsama_testi(p)

    print("dt (ms)   hata")
    for dt, h in zip(dt_listesi, hatalar):
        print(f"  {dt:.4f}   {h:.3e}")

    print("\nyakinsama derecesi (ardisik ciftler icin):")
    for i in range(len(dt_listesi) - 1):
        oran = hatalar[i] / hatalar[i + 1]
        derece = np.log2(oran)   # dt yariya inince hata kac kat kuculuyor (log2)
        print(f"  dt {dt_listesi[i]:.4f} -> {dt_listesi[i+1]:.4f} : derece ~ {derece:.2f}")

    plt.figure(figsize=(5.5, 4.5))
    plt.loglog(dt_listesi, hatalar, "o-", label="RK4 hatasi")

    # karsilastirma icin egim=4 cizgisi ciziyorum
    dt_arr = np.array(dt_listesi)
    C = hatalar[-1] / (dt_arr[-1] ** 4)
    plt.loglog(dt_arr, C * dt_arr**4, "--", color="gray", label="egim=4 (beklenen)")

    plt.xlabel("dt (ms)")
    plt.ylabel("hata")
    plt.title("RK4 dt-yakinsama testi")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    import os
    klasor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
    os.makedirs(klasor, exist_ok=True)
    plt.savefig(os.path.join(klasor, "dt_yakinsama.png"), dpi=150)
    print("\ngrafik kaydedildi: figs/dt_yakinsama.png")
