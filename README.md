# gumushane_eem_eesec440.
# EESEC 440 English for EEE on DL

I. Artificial Neural Networks
Yapay sinir ağları (YSA), girdi-çıktı verilerini son derece karmaşık bir şekilde ilişkilendirebilen son derece güçlü yapılardır. Yeni nesil Grafik İşlem Birimlerinin (GPU) son on yılda sağladığı hesaplama gücündeki artış, YSA'nın çeşitli çalışma alanlarında (örneğin Bilgisayarla Görü, konuşma tanıma, Doğal Dil İşleme, güç aktarımı gibi) birçok mühendislik problemini çözmesini sağlamıştır [1]. , vb). İhtiyaçlara göre farklı uygulamalar için tasarlanmış farklı YSA çeşitleri vardır. Çok katmanlı Algılayıcı (MLP), bir YSA'nın iki ana işlevi olan regresyon ve sınıflandırma problemlerinde yaygın olarak kullanılan en çok tercih edilen ağ türlerinden biridir.
*A. Çok Katmanlı Algılayıcı (MLP)*
Kurs boyunca MLP'yi ayrıntılı olarak inceleyeceğiz. Regresyon ve sınıflandırma örnekleri, her ikisi de denetimli öğrenme. Örnek bir MLP ağ yapısı için lütfen Şekil 1'e bakın.
![122469761-fb816400-cfc5-11eb-876d-3af765f73169](https://user-images.githubusercontent.com/83355659/122642247-182ab280-d112-11eb-8d4c-cc74e7e9ea8c.png)
Şekil 1: Çok girişli tek çıkışlı MLP.
*Exclusive OR (XOR) Problemi*
![122469163-4babf680-cfc5-11eb-86a0-534f9aef5e9b](https://user-images.githubusercontent.com/83355659/122642290-57f19a00-d112-11eb-9df1-3da53eb7cbae.jpg)
Şekil 2: Bir MLP ağı XOR problemini öğrenecek.
![122468881-f374f480-cfc4-11eb-923a-8d33ee8667cf](https://user-images.githubusercontent.com/83355659/122642310-7fe0fd80-d112-11eb-825c-d4192d8fa764.jpg)
Şekil 3: XOR problemi için, on altı nörondan oluşan bir gizli katmana sahip bir MLP kullanıyoruz.
![122468930-038cd400-cfc5-11eb-9a71-c9c050e5ad3d](https://user-images.githubusercontent.com/83355659/122642317-896a6580-d112-11eb-83d4-064039b01818.jpg)
Şekil 4: Ağ parametrelerini güncellemek için hata geri yayma algoritması (örn. Gradient Descent) kullanılır.
*Çok Katmanlı bir Yapay Sinir Ağının Ayarlanabilen Parametre (Ağırlık) Sayısı*
![122469066-2ae3a100-cfc5-11eb-979b-86eb186d37c4](https://user-images.githubusercontent.com/83355659/122642342-b159c900-d112-11eb-88c1-2c60bd9d2a6c.jpg)
Şekil 5: Tek girişli, gizli katmanda üç nöronlu ve tek çıkışlı bir MLP ağı. Bu ağ 2 : 3 : 1 konfigürasyona sahiptir. 2 : 3 : 1 besleme hattının şekil 2 : 3 : 1 beslenmesi, ana hatlarında, ana tesisatlarında ve ön cephelerinde bir ön cephelidir (yuvarlaklarları). parametre sayısı2 13 olarakta w1, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13. uygulanabilena göre ayarlanabilen parametre (ağırlık) hem çizerek, hem de kısa bir şekilde hesaplanabilir. Formülü size de biraz üretimden kaynaklanır. Giriş, çıkış ve bir katmandan oluşan bir yapay ağındaki toplam ayardan oluşan parametre (ağırlık - ağırlık) bulan formül:
n = (n1 x n2 + n2) + (n2 x n3 + n3)
Burada n 1 giriş noktasındaki sayıların sayısı, katmandaki sayısı ve sayısı 3 anadaki ağırlıktaki ana gövdenin giriş kısmındaki sayı sayısı (-w) geneldir. Yukardaki formülde uygun terimler ortak paratez alınırsa, hazırlanır:
n = n2(n1 + 1) + n3(n2 + 1)
Şekil 5'de verilen 2 : 3 : 1 kalıplu formülde test olacaksa n = 3(2 + 1) + 1(3 + 1) = 3 x 3 + 1 x 4 = 13 onaylamış olabilir.
![122469282-71d19680-cfc5-11eb-87c1-1a834005ce0d](https://user-images.githubusercontent.com/83355659/122642407-f847be80-d112-11eb-81a7-3262360f4db5.jpg)
2 Boyutlu Sınıflandırma Problemi
![122469353-86159380-cfc5-11eb-9ec1-92afdc84ff88](https://user-images.githubusercontent.com/83355659/122642425-04cc1700-d113-11eb-931c-39806d88248a.jpg)
Şek. 6: 2 boyutlu modelleme probleminde en üst düzeyde öğretmek.
En Küçük Kareler
![122469391-93328280-cfc5-11eb-8e56-4cb25022fbc8](https://user-images.githubusercontent.com/83355659/122642440-16152380-d113-11eb-8f84-fe2b408fbdf5.png)
Şekil 7: Bir boyutlu optimizasyon (veya regresyon) probleminin görselleştirilmiş hali.
*B. Evrişimsel Sinir Ağı (CNN)*
Dersin son haftalarında görseller üzerinde sınıflandırma örneği çalışabiliriz. Kaggle'da kedi-köpek görüntü deposu. [2]'de verilen öğreticiyi takip edeceğiz. Kedi ve köpek görüntülerinin CNN aracılığıyla sınıflandırılması için lütfen Şekil 8 ve 9'a bakın.
<img width="1153" alt="122469484-afceba80-cfc5-11eb-9a4a-70c842359fcc" src="https://user-images.githubusercontent.com/83355659/122642590-eca8c780-d113-11eb-8260-513f4b87a5b5.png">
Şekil 8: CNN aracılığıyla kedi ve köpek görüntülerinin sınıflandırılması.
![122469518-b826f580-cfc5-11eb-9f65-dd3137e880c8](https://user-images.githubusercontent.com/83355659/122642643-237edd80-d114-11eb-93b1-bdecc7c89f43.gif
![122469518-b826f580-cfc5-11eb-9f65-dd3137e880c8](https://user-images.githubusercontent.com/83355659/122642738-98eaae00-d114-11eb-8485-14c352164274.gif)
Şekil 9: Derin öğrenme ile kedi ve köpeklerin sınıflandırılması.

*Referanslar*
[1] T. Guillod, P. Papamanolis ve JW Kolar, "Yapay Sinir Ağı (YSA) Tabanlı Hızlı ve Doğru İndüktör Modelleme ve Tasarım", IEEE Open Journal of Power Electronics, cilt. 1, pp. 284-299, 2020, doi: 10.1109/OJPEL.2020.3012777. [2] "Köpek ve kedi fotoğrafları nasıl sınıflandırılır (%97 doğrulukla)." https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/ . Erişim tarihi: 2021-3-10.

