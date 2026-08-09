# Hedef

GaussCapture'a sabit-kamera 4D desteği uçtan uca çalışır halde girecek: kullanıcı
telefonu bir desteğe koyup dört fazlı `F_bullet` protokolüyle kendi yüzünü 1-3 saniye
çekecek, çıkan yakalamayı laptopta hazırlayıp Colab/kiralık GPU'da eğitecek, ve sonucu
tarayıcıda ±25° koniye kilitli bir 4D Gaussian splat olarak izleyecek. Uygulama gerçek
bir release anahtarıyla imzalanmış APK olarak GitHub Release'te indirilebilir olacak ve
statik bir sayfa bunu checksum'ıyla birlikte sunacak.

# Görevler

- [x] 1. 4D mimarisi: lisans-temiz yöntem, iki fazlı çekim, SE(3) scaffold kararı
- [x] 2. Çekirdek kod: deformation trainer, `.g4d` export, 4D viewer, faz ayırma, dinamik maske
- [x] 3. Matematik onarımı: `so3.py` broadcast, smallest-three quantisation, rijit hareket
- [x] 4. Android: dört fazlı protokol, gerçek launcher ikonu, `flutter build apk --release`
- [x] 5. Release imzalama zinciri (keystore + `key.properties` + Gradle + gitignore)
- [x] 6. Lisans kapısını `git ls-files` üzerinden çalışacak şekilde sağlamlaştır
- [ ] 7. Şüpheci sondanın bulgularını kapat (blocker'lar bitti; minor'lar sürüyor)
- [ ] 8. Torch'suz temiz venv'de CI ortamını birebir sına — `pip install -e ".[dev]"` + pytest
- [ ] 9. APK'yi son kodla yeniden build et, imzayı `apksigner` ile doğrula, SHA-256 al
- [ ] 10. Commit + push; GitHub CI'ın yeşile döndüğünü doğrula
- [ ] 11. GitHub Release oluştur, APK'yi asset olarak yükle
- [ ] 12. `site/` içindeki checksum'ı gerçek APK'ninkiyle güncelle, Pages yayınını doğrula
- [ ] 13. APK'yi telefona kur, kullanıcı çekim yapsın, sonucu tarayıcıda izle

# Bitti sayılma ölçütü

- `.venv/bin/ruff check .` → `All checks passed!`
- `.venv/bin/python -m pytest -q` → hepsi geçiyor (torch kurulu ortamda)
- Torch'suz temiz venv'de `pip install -e ".[dev]"` sonrası pytest → toplama çökmüyor,
  torch gerektiren testler `skipped` olarak geçiyor, geri kalanı `passed`
- Lisans kapısı: `git ls-files` taraması hiçbir yasaklı string bulmuyor
- GitHub Actions'ta `CI` yeşil (test matrisi + licence gate + capture protocol JVM)
- `apksigner verify --print-certs` → `CN=Enes Oz` (asla `CN=Android Debug`)
- `https://en970.github.io/gausscapture/download.html` APK'yi ve doğru SHA-256'yı gösteriyor
- Kullanıcı telefonda çekim yapıp `.g4d` sonucunu tarayıcıda oynatabiliyor

# Sınırlar

- **Lisans:** Inria'nın ticari olmayan 3DGS bileşenlerinin adları (tam liste
  `.github/workflows/ci.yml` içindeki `FORBIDDEN` deseninde ve `docs/DEPENDENCIES.md`'de)
  `docs/`, `site/`, `.github/` dışında hiçbir **takip edilen** dosyada geçemez — bu dosya
  dahil, ki nitekim ilk yazımında geçtiği için CI'ı kırdı. Rasterizer gsplat (Apache-2.0).
- **Dürüstlük:** Sabit kameradan sınırsız serbest bakış vaat eden hiçbir metin yazılmayacak —
  kod yorumu, docs, site, viewer arayüzü dahil. ±25° koni ürünün kendisidir.
- **Keystore:** `app/android/gausscapture-release.jks` ve `key.properties` asla commit edilmez.
  Yedeği `~/Desktop/GaussCapture-release-imza/`.
- **Testler:** Bir başarısızlığı gizlemek için test gevşetilmez, assertion silinmez, tolerans
  genişletilmez. Test gerçekten yanlışsa gerekçesiyle düzeltilir.
- **Donanım:** Bu laptop M2 Air, CUDA yok. gsplat CUDA-only. Eğitim Colab/kiralık GPU'da;
  laptopta yakalama, hazırlık, export ve izleme çalışır.
- **Kullanıcı kararı:** APK'yi telefona kurmayı kullanıcı üstlendi; sorulmadan kurulmaz.
