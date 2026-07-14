# rejim_haritasi.py
# Burada iki parametreyi (eta ve tau_d) ayni anda degistirip sistemin
# nasil davrandigina (sabit nokta mi, periyodik mi, karisik mi) bakiyorum.
# Ayrica Lyapunov ustelini hesaplayarak kaotik olup olmadigini kontrol
# ediyorum (Benettin yontemi - hocamin onerdigi klasik yontem).
#
# ONEMLI: Bir ara r cok sivri (keskin) ates etme rejimlerinde RK4 sayisal
# olarak patliyordu (NaN degerler cikiyordu), ben de bunu fark etmeden
# ilk basta "sabit nokta" olarak etiketlemistim - bu yanlismis, cunku
# NaN karsilastirmalari hep False donuyor, bos liste geliyor, ben de
# bunu "degisim yok, sabit nokta" saniyordum. Asagida bunu duzelttim,
# artik NaN cikan yerleri ayri ("iraksadi") diye isaretliyorum.

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import get_params, baslangic_durumu, turevler


def rk4_adim(xi, r_gecmis, dt, p):
    k1 = turevler(xi, r_gecmis, p)
    k2 = turevler(xi + 0.5 * dt * k1, r_gecmis, p)
    k3 = turevler(xi + 0.5 * dt * k2, r_gecmis, p)
    k4 = turevler(xi + dt * k3, r_gecmis, p)
    return xi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simule_et_basit(p, x0, t_max, dt):
    n_adim = int(t_max / dt)
    X = np.zeros((n_adim + 1, len(x0)))
    X[0] = x0
    gecikme_adim = int(round(p["tau_d"] / dt)) if p["tau_d"] > 0 else 0

    for i in range(n_adim):
        xi = X[i]
        if gecikme_adim == 0:
            r_gec = xi[0]
        else:
            j = i - gecikme_adim
            r_gec = x0[0] if j < 0 else X[j, 0]
        X[i + 1] = rk4_adim(xi, r_gec, dt, p)

    return X


def tepe_noktalarini_bul(dizi):
    # basit bir "lokal maksimum" bulucu
    tepeler = []
    for i in range(1, len(dizi) - 1):
        if dizi[i] > dizi[i - 1] and dizi[i] > dizi[i + 1]:
            tepeler.append(dizi[i])
    return np.array(tepeler)


def rejim_belirle(r_son, tol=0.02):
    # once NaN/Inf var mi diye bakiyoruz - varsa sistem iraksamis demektir
    if not np.all(np.isfinite(r_son)):
        return "IRAKSADI", -1

    tepeler = tepe_noktalarini_bul(r_son)
    if len(tepeler) < 4:
        return "sabit_nokta", 0

    son_tepeler = tepeler[-40:] if len(tepeler) > 40 else tepeler

    # farkli tepe degerlerini gruplandiriyoruz (birbirine yakin olanlar ayni grup)
    gruplar = []
    for deger in son_tepeler:
        eslesti = False
        for g in gruplar:
            if abs(deger - g) < tol * max(1.0, abs(g)):
                eslesti = True
                break
        if not eslesti:
            gruplar.append(deger)

    n = len(gruplar)
    if n <= 6:
        return f"periyot-{n}", n
    return "karisik/kuazi-periyodik", n


def rejim_haritasi_hesapla(eta_listesi, tau_d_listesi, t_max=2200.0, gecici=1600.0, dt=0.03):
    kodlar = np.zeros((len(eta_listesi), len(tau_d_listesi)), dtype=int)
    iraksama_sayisi = 0

    for i, eta in enumerate(eta_listesi):
        for j, tau_d in enumerate(tau_d_listesi):
            p = get_params()
            p["eta"] = eta
            p["tau_d"] = tau_d
            x0 = baslangic_durumu(p)

            X = simule_et_basit(p, x0, t_max, dt)
            t = np.arange(len(X)) * dt
            r_son = X[t > gecici, 0]

            etiket, kod = rejim_belirle(r_son)
            kodlar[i, j] = kod
            if kod == -1:
                iraksama_sayisi += 1

    if iraksama_sayisi > 0:
        print(f"  (not: {iraksama_sayisi} noktada RK4 iraksadi, dt kucultulmesi lazim olabilir)")

    return kodlar


def lyapunov_hesapla(p, x0, t_max=12000.0, dt=0.05, yeniden_normlama_araligi=5.0, gecici=6000.0):
    # Benettin yontemi: iki birbirine cok yakin baslangicla ayni sistemi
    # calistirip aralarindaki mesafenin zamanla nasil buyudugune bakiyoruz.
    # Eger mesafe surekli buyuyorsa (ustel olarak) bu kaotik demek.
    n_adim = int(t_max / dt)
    gecikme_adim = int(round(p["tau_d"] / dt)) if p["tau_d"] > 0 else 0
    normlama_her_kac_adim = max(1, int(round(yeniden_normlama_araligi / dt)))
    gecici_adim = int(round(gecici / dt))

    d0 = 1e-8
    X1 = np.zeros((n_adim + 1, len(x0)))
    X2 = np.zeros((n_adim + 1, len(x0)))
    X1[0] = x0
    X2[0] = x0.copy()
    X2[0][0] += d0   # r yonunde ufak bir sapma veriyoruz

    toplam_log = 0.0
    kac_kere_normlandi = 0

    for i in range(n_adim):
        if gecikme_adim == 0:
            r1 = X1[i][0]
            r2 = X2[i][0]
        else:
            j = i - gecikme_adim
            r1 = x0[0] if j < 0 else X1[j, 0]
            r2 = x0[0] if j < 0 else X2[j, 0]

        X1[i + 1] = rk4_adim(X1[i], r1, dt, p)
        X2[i + 1] = rk4_adim(X2[i], r2, dt, p)

        if not (np.all(np.isfinite(X1[i + 1])) and np.all(np.isfinite(X2[i + 1]))):
            return np.nan   # iraksadi, guvenilir sonuc yok

        adim_no = i + 1
        if adim_no % normlama_her_kac_adim == 0 and adim_no > gecici_adim:
            fark = X2[adim_no] - X1[adim_no]
            mesafe = np.linalg.norm(fark)
            if mesafe > 0:
                toplam_log += np.log(mesafe / d0)
                kac_kere_normlandi += 1
                # yeniden normluyoruz ki sapma cok buyumesin
                olcek = d0 / mesafe
                X2[adim_no] = X1[adim_no] + fark * olcek
                if gecikme_adim > 0:
                    alt = max(0, adim_no - gecikme_adim)
                    gecmis_fark = X2[alt:adim_no + 1] - X1[alt:adim_no + 1]
                    X2[alt:adim_no + 1] = X1[alt:adim_no + 1] + gecmis_fark * olcek

    if kac_kere_normlandi == 0:
        return np.nan

    lambda_ms = toplam_log / (kac_kere_normlandi * yeniden_normlama_araligi)
    return lambda_ms * 1000.0   # 1/saniye cinsinden donduruyoruz


if __name__ == "__main__":
    print("Lyapunov ustelini hesapliyorum (biraz zaman alabilir)...")
    print("(not: astrosit kismi cok yavas (~7 saniyelik) bir zaman olcegine")
    print(" sahip, o yuzden Hopf esigine yakin gecikmelerde deger hala")
    print(" gecici rejimden etkilenmis olabilir, tam guvenilir degil)")

    tau_d_degerleri = [8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
    lyapunov_sonuclari = []

    for tau_d in tau_d_degerleri:
        p = get_params()
        p["tau_d"] = tau_d
        x0 = baslangic_durumu(p)
        lam = lyapunov_hesapla(p, x0)
        lyapunov_sonuclari.append(lam)
        not_yaz = " (iraksadi)" if np.isnan(lam) else ""
        print(f"  tau_d = {tau_d} ms  ->  lambda_max = {lam:.4f}{not_yaz}")

    print("\nrejim haritasini hesapliyorum (bu da biraz surer)...")
    eta_degerleri = np.linspace(-2.5, -0.5, 8)
    tau_d_degerleri_harita = np.linspace(4.0, 16.0, 10)
    kodlar = rejim_haritasi_hesapla(eta_degerleri, tau_d_degerleri_harita)

    fig, eksen = plt.subplots(1, 2, figsize=(12, 4.5))

    gosterim = np.ma.masked_where(kodlar < 0, kodlar)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="red")

    im = eksen[0].imshow(gosterim, origin="lower", aspect="auto",
                          extent=[tau_d_degerleri_harita[0], tau_d_degerleri_harita[-1],
                                  eta_degerleri[0], eta_degerleri[-1]],
                          cmap=cmap)
    eksen[0].set_xlabel("tau_d (ms)")
    eksen[0].set_ylabel("eta")
    eksen[0].set_title("Rejim haritasi (kirmizi = sayisal olarak iraksadi)")
    plt.colorbar(im, ax=eksen[0], label="periyot sayisi")

    eksen[1].plot(tau_d_degerleri, lyapunov_sonuclari, "o-", color="darkred")
    eksen[1].axhline(0, color="gray", linestyle="--")
    eksen[1].set_xlabel("tau_d (ms)")
    eksen[1].set_ylabel("lambda_max (1/s)")
    eksen[1].set_title("Lyapunov usteli (Benettin)")

    plt.tight_layout()

    import os
    klasor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
    os.makedirs(klasor, exist_ok=True)
    plt.savefig(os.path.join(klasor, "rejim_haritasi.png"), dpi=150)
    print("\ngrafik kaydedildi: figs/rejim_haritasi.png")
