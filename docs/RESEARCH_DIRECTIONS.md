# Usulan eksperimen Checkpoint 1

Dokumen ini mencatat hipotesis awal dari eksperimen Flowers102 dengan konfigurasi tetap. Status dan hasil selective fusion yang dikerjakan sesudahnya dibahas di [notebook Checkpoint 2](../notebooks/02_selective_fusion_checkpoint2.ipynb).

## A. Confidence-based Selective Fusion

### Temuan yang memotivasi

Linear dan LoRA sama-sama benar pada 198/200 gambar, tetapi tidak berbagi gambar error. Dua error per metode belum cukup untuk menyimpulkan error diversity yang sistematis. Probability averaging keduanya benar pada 200/200 gambar validation yang sudah dianalisis. NLL averaging sekitar 0.0734, lebih tinggi daripada LoRA sendiri sekitar 0.0410. Tujuan mencakup kualitas probabilitas dan biaya, selain akurasi. Temperature-scaled Linear belum diuji untuk memeriksa pengaruh skala confidence terhadap selisih NLL.

### Mekanisme yang diusulkan

Jalankan Linear terlebih dahulu. Jika calibrated confidence memenuhi aturan penerimaan, gunakan prediksinya. Untuk input lain, jalankan LoRA lalu gabungkan prediksi kedua metode. Bandingkan probability averaging dan logit averaging sebagai ablation aturan fusion, karena NLL keduanya berbeda pada eksplorasi validation saat ini. Temperature tiap model dan aturan routing dipilih pada calibration data yang terpisah dari evaluasi akhir. Kedua metode menjalankan ViT penuh. Dengan proporsi fallback r, rata-rata jumlah forward pass adalah 1+r. Ini bukan rasio latency. Dua forward pass pada kasus fallback, biaya gate, serta biaya memuat dan menyimpan kedua model harus masuk pengukuran.

Pertanyaan penelitian: apakah fusion heterogen selektif memberi trade-off kualitas dan latency yang lebih baik daripada calibrated single model, ensemble seed dengan metode sama, serta always-on fusion?

### Pembanding dan pengujian

Bandingkan Linear dan LoRA sebelum serta sesudah temperature scaling, ensemble beberapa seed dari metode yang sama, always-on equal averaging, routing acak pada anggaran sama, confidence gate tanpa calibration, dan gate dengan calibration. Ensemble classifier Linear dengan backbone identik dapat berbagi satu forward backbone saat inferensi. Kontrol ini memeriksa apakah manfaat memerlukan metode heterogen atau cukup variasi seed dan calibration. Pembuatan kontrol seed merupakan eksperimen lanjutan yang memerlukan training.

Ukur accuracy, NLL, Brier, risk-coverage, proporsi input yang memakai model kedua, latency end-to-end dan memory residency. Bandingkan pada anggaran inferensi yang sama dan laporkan biaya training kontrol seed. Hasil akhir memakai test setelah aturan dikunci, dengan beberapa seed untuk menilai kestabilan. Validation 200 gambar sudah diamati, sehingga penggunaannya untuk calibration harus dinyatakan sebagai development. Cakupan 90 kelas juga harus dicatat. Jangan memilih aturan memakai label test atau hanya lima contoh error yang ditampilkan.

Sebelum evaluasi akhir, tetapkan batas penurunan kualitas yang dapat diterima dan target penghematan resource. Selective fusion diteruskan jika memberi trade-off quality-latency yang berguna terhadap calibrated single model serta always-on fusion. Jika kontrol tersebut menyamai atau mendominasi hasilnya, kompleksitas routing tidak mendapat dukungan empiris.

### Posisi terhadap literatur

Mai et al. sudah memeriksa error diversity dan ensemble. Caption Figure 4 menyebut majority vote, sedangkan rincian Figure 4 pada Appendix C menjelaskan average logits. Notebook membandingkan average logits dan average probabilitas secara eksplisit. EulerLoRA menghasilkan keragaman prediktif melalui variasi stokastik komponen LoRA. Lotfi et al. membandingkan ensembling dan routing untuk expert PEFT pada LLM. Usulan ini menguji cascade heterogen Linear-LoRA untuk klasifikasi visual dengan batas pemanggilan expert serta evaluasi probabilitas dan latency aktual. Kontribusi yang diuji adalah manfaat heterogenitas terhadap kontrol calibration dan variasi seed pada anggaran inferensi yang sama. Routing confidence sendiri tidak diposisikan sebagai kontribusi baru. Keuntungan mekanismenya masih perlu dibuktikan.

## B. Class-preserving Sample Weighting

### Temuan yang memotivasi

Training hanya memiliki 5-10 gambar per kelas. Ini membuat pengurangan kontribusi beberapa sampel berpotensi berdampak besar pada kelas tertentu. Tidak ada bukti dari analisis sekarang bahwa label Flowers102 yang digunakan berisik, atau bahwa error saat ini disebabkan oleh imbalance.

### Mekanisme yang diusulkan

Gunakan loss atau indikator kesulitan yang berasal dari proses training. Ubah indikator tersebut menjadi bobot relatif di dalam kelas, lalu beri batas minimum total kontribusi setiap kelas. Utamakan soft weighting agar kelas kecil tetap menyumbang gradien. Informasi label yang sengaja dikorupsi hanya dipakai untuk audit eksperimen, tidak diberikan sebagai fitur atau mask oracle pada metode.

Pertanyaan penelitian: apakah class-preserving sample weighting memperbaiki recall kelas kecil ketika label noise dan ketimpangan kelas muncul bersamaan?

### Pembanding dan pengujian

Pilot menggunakan dua faktor: noise 0% versus 20% dan distribusi asli versus long-tail terkontrol dengan target rasio head/tail sekitar 5, dengan minimal satu sampel per kelas. Korupsi hanya dilakukan pada training. Bandingkan LoRA biasa, class weighting, global noise weighting, serta within-class weighting dengan contribution floor. Sertakan kontrol SSF untuk memeriksa apakah manfaat berasal dari loss umum atau interaksi dengan LoRA. Ukur macro-F1, balanced accuracy, recall head/tail dan NLL pada evaluasi bersih. Jalankan beberapa seed setelah pilot integrasi.

### Posisi terhadap literatur

Ma, Pan dan Wei (ICMR 2026) membahas robustness PEFT pada noisy labels melalui gradient-loss decoupling. Usulan kita berfokus pada kontribusi kelas kecil ketika weighting menangani noise dan class imbalance secara bersama. Metadata dan abstract paper ini telah diperiksa, tetapi rincian full text masih perlu ditelaah sebelum mengunci perbedaan mekanisme terhadap seluruh intervensi mereka. Arah ini membutuhkan training lebih banyak daripada analisis routing atas model yang sudah selesai.

## C. Resource-aware PEFT Configuration Selection

### Temuan yang memotivasi

Ketiga metode memiliki akurasi sama dan NLL berbeda. SSF melatih parameter lebih sedikit daripada LoRA, tetapi peak GPU allocated yang tercatat pada run SSF lebih besar. Catatan resource tersebut berasal dari kondisi node bersama dan membutuhkan profiling terkontrol sebelum diinterpretasikan sebagai perbedaan intrinsik metode. Batas host RAM cgroup 16 GiB juga memengaruhi kelayakan startup, walaupun perangkat GPU adalah A100 80 GB.

### Mekanisme yang diusulkan

Bentuk kandidat konfigurasi Linear, LoRA dan SSF dengan effective batch tetap 64, precision FP32, serta pilihan microbatch yang dinyatakan sebelumnya. Profiling singkat menghasilkan estimasi peak host RAM, GPU allocated/reserved, serta waktu update. Saring konfigurasi yang memenuhi batas resource, lalu pilih dengan kualitas validation dan biaya. Aturan pemilihan meminimalkan NLL dengan kendala accuracy minimum dan budget resource yang ditentukan sebelum evaluasi.

### Pembanding dan pengujian

Bandingkan pemilihan berdasarkan accuracy saja, parameter count saja, resource saja dan aturan gabungan. Biaya profiling dan pencarian ikut dihitung. Profiling menggunakan warmup, jumlah update dan kondisi pengukuran yang sama. Konfigurasi baru membutuhkan training kualitas yang sesuai sebelum skornya dibandingkan. Uji pada lebih dari satu task serta beberapa seed dan laporkan kelayakan, NLL, accuracy, biaya total serta keberhasilan menyelesaikan run. Reproduksi resource tidak mengubah batas server atau menyentuh proses pengguna lain.

### Posisi terhadap literatur

Appendix A paper utama menyatakan bahwa studi tidak berfokus pada memory usage dan FLOPs. Khan dan Siddique mengevaluasi PET dengan parameter budget dan calibration pada klasifikasi chest X-ray multimodal. Usulan ini menguji pemilihan konfigurasi visual PEFT berdasarkan kelayakan host/GPU aktual, biaya profiling dan kualitas probabilitas pada satu pipeline yang sama. Bentuk kontribusinya adalah evaluasi terkontrol beserta kebijakan pemilihan konfigurasi.

## Referensi pembanding

- Mai et al., CVPR 2025: https://arxiv.org/abs/2409.16434
- Anumasa dan Liu, EulerLoRA, arXiv 2026: https://arxiv.org/abs/2608.01142
- Lotfi et al., Trade-offs in Ensembling, Merging and Routing Among Parameter-Efficient Experts, arXiv 2026: https://www.microsoft.com/en-us/research/publication/trade-offs-in-ensembling-merging-and-routing-among-parameter-efficient-experts/
- Ma, Pan dan Wei, ICMR 2026: https://doi.org/10.1145/3805622.3810828
- Khan dan Siddique, ICCIT 2025 sebagaimana dinyatakan pada arXiv record: https://arxiv.org/abs/2512.21508
