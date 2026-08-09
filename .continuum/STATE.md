# Şu an

Şüpheci sondanın bulguları kapatıldı; sıra bunların gerçekten tuttuğunu **torch'suz temiz
bir sanal ortamda** kanıtlamakta — CI'ın kurduğu ortam bu ve daha önce orada sıfır test
çalışıyordu.

# Sıradaki adım

Temiz bir venv kur ve CI ortamını birebir taklit et:

    python3 -m venv /tmp/gc-ci && /tmp/gc-ci/bin/pip install -e ".[dev]"
    /tmp/gc-ci/bin/python -m pytest -q

Beklenen: toplama (collection) çökmüyor, torch gerektiren testler `skipped`, geri kalanı
`passed`. Şüpheci ajan bu ortamda daha önce `3 failed, 348 passed, 1 skipped, 17 errors`
ölçmüştü; hata kaynağı `tests/test_pipeline4d.py`'nin `prepared`/`packaged`/`trained`
fixture'larının `deform/bundle.py:32`'deki modül seviyesi `import torch`'a ulaşmasıydı.
`prepare4d.py:62` artık torch'suz `deform.init4d` kullanıyor; doğrulanacak olan, aynı
yolun `bundle.py` üzerinden hâlâ torch'a düşüp düşmediği.

Sonrasında sırasıyla: APK'yi son kodla yeniden build + `apksigner` doğrulaması, commit +
push, CI'ın yeşile döndüğünü izleme, GitHub Release + APK asset, `site/` checksum
güncellemesi, Pages yayını.

# Tamamlananlar

- 4D mimarisi kararlaştırıldı: gsplat üstünde lisans-temiz yeniden yazım, dört fazlı
  `F_bullet` çekimi, SE(3) scaffold hareket temsili
- Çekirdek kod yazıldı: `recon/deform/` (trainer), `export/scene4d.py` (`.g4d`),
  `report/viewer_4d.js`, `ingest/phases.py`, `recon/dynamic_mask.py`, `pose/cone.py`,
  `pose/fixed.py`, `recon/prepare4d.py`, `recon/fit4d.py`
- Altı yeni CLI komutu: `mask`, `prep4d`, `colab4d`, `train4d`, `export4d`, `viewer4d`
- Matematik onarıldı: `so3.quat_rotate` rank broadcast; smallest-three quantisation'da
  sıfırın kodlanabilir olmaması (identity 0.137° sapıyordu → tam 0)
- İki test hatası düzeltildi: rijit hareket testi ölü satırla bozuluyordu; snorm16 testi
  `arccos` metriği yüzünden bir kat büyük okuyordu
- Android: dört fazlı protokol (`PhaseMachine`, `Stillness`, `Arc`), gerçek launcher
  ikonu (amber, zamanda kaymış elipsler, adaptive icon dahil), `flutter analyze` temiz,
  `flutter build apk --release` başarılı
- Release imzalama zinciri: 4096-bit keystore, `key.properties`, Gradle signingConfig,
  `.gitignore`. APK artık `CN=Enes Oz` ile imzalı (önceden `CN=Android Debug` idi)
- Lisans kapısı `git ls-files` üzerinden çalışacak şekilde sağlamlaştırıldı (gsplat'in
  kendi kaynağı yasaklı dizgeleri kullanılmayan bir sarmalayıcıda barındırıyor)
- Statik site yazıldı: `site/index.html`, `download.html`, sıkı CSP, sıfır harici istek
- Dört blocker kapatıldı: `test_deform_reference.py` torch guard'ı, Colab defterinin
  yanlış importu, `fit4d`'in `GsplatRasterizer`'ı hiç kullanmaması, `prepare4d`'in torch'a
  bağımlılığı
- `pyproject.toml`'a `train4d` extra'sı eklendi (üç hata mesajı bu dizgeyi söylüyordu)
- Testler 211 → 461, `ruff check .` temiz

# Bilinmesi gerekenler

- **Oturum limiti üç kez vurdu** (00:40, 15:40, 01:50 sıfırlamaları). Workflow'ların
  doğrulama fazları hep bu yüzden düştü. Uzun workflow yerine kısa, odaklı workflow'lar
  veya doğrudan kendi ölçümüm daha güvenli.
- **"417/461 passed" rakamı yanıltıcıydı**: yalnızca torch kurulu olan yerel venv'de
  geçerli. CI ortamı `.[dev]` kuruyor ve orada toplama çöküyordu. Yeşil sayıya, hangi
  ortamda ölçüldüğünü sormadan güvenme.
- **gsplat CUDA-only.** Bu laptop M2 Air; gsplat kurulamaz, kurulmaya çalışılmamalı.
  CPU tarafı `ReferenceRasterizer` ile test edilir, gerçek eğitim Colab/kiralık GPU'da.
- `ReferenceRasterizer` O(N·H·W) ve `(N, H*W, 2)` tensörü materyalize ediyor: 8.000
  Gaussian / 128×128'de 5,9 s ve 1,05 GB. Gerçek çözünürlükte kullanılamaz — sadece test.
- **Keystore** `app/android/gausscapture-release.jks`, parola `key.properties` içinde,
  yedeği `~/Desktop/GaussCapture-release-imza/`. Repoya asla girmez. Kaybolursa kurulu
  uygulama bir daha güncellenemez.
- Viewer'ın sıralama worker'ında bulunan bir hata: premultiplied over-blend arkadan öne
  sıralama ister, worker önden arkaya veriyordu. Onarım ajanına bildirildi; düzeltildiğini
  doğrula.
- `.g4d` kontratı (Python yazıcı ↔ JavaScript okuyucu) **çalıştırılarak** doğrulandı ve
  temiz çıktı; quaternion konvansiyonu `(w,x,y,z)` iki tarafta tutarlı.
- Kullanıcı APK'yi telefona kendisi yüklemek istedi; sorulmadan kurulmayacak. Telefon
  (Galaxy S22, SM-S901E) adb ile bağlı ve uygulama şu an **kurulu değil**.

# Son güncelleme

10 Ağustos 2026, 00:35
