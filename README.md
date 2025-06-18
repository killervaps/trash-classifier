# **Trash Classifier**

## **Identitas Anggota Kelompok**
| Nama | NRP |
|-----------|-----------|
| Samuel Steve Mulyono | 5025231197 |
| Naswan Nashir Ramadhan | 5025231246 |
| Valensio Arvin Putra Setiawan | 5025231273 |

Link Dataset : [TrashNet](https://huggingface.co/datasets/garythung/trashnet)

## Informasi Proyek
Program ini dirancang untuk mengklasifikasikan jenis sampah untuk daur ulang menggunakan Convolutional Neural Network (CNN), sebuah pendekatan deep learning yang unggul dalam mengenali fitur visual dari gambar. Implementasi spesifik melibatkan dataset TrashNet yang terdiri dari 2.527 gambar terbagi dalam 6 kelas (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`). Proyek ini memanfaatkan teknik transfer learning dengan model pre-trained seperti MobileNetV2, EfficientNetB0, dan ResNet50, dilengkapi dengan augmentasi data untuk meningkatkan performa. Tahapan meliputi persiapan data dengan ImageDataGenerator, pelatihan model menggunakan pendekatan dua tahap (head training dan fine-tuning), pengujian prediksi ensemble pada gambar yang diunggah, serta evaluasi menggunakan confusion matrix dan classification report. Hasilnya diarahkan untuk implementasi berbasis web menggunakan Streamlit, memungkinkan prediksi real-time berdasarkan gambar yang diunggah, sesuai dengan tujuan otomatisasi daur ulang.

---

## Pembahasan

### Tahap 1: Persiapan Lingkungan dan Data Generator
  ```python
  import tensorflow as tf
  from tensorflow.keras import layers, models, optimizers
  from tensorflow.keras.preprocessing.image import ImageDataGenerator
  from tensorflow.keras.callbacks import EarlyStopping
  import numpy as np
  import random
  import os

  # Set seeds for reproducibility
  SEED = 42
  tf.random.set_seed(SEED)
  np.random.seed(SEED)
  random.seed(SEED)

  # Augmentation & generators
  def get_data_generators(img_size):
      train_datagen = ImageDataGenerator(
          rescale=1./255,
          rotation_range=20,
          width_shift_range=0.1,
          height_shift_range=0.1,
          shear_range=0.1,
          zoom_range=0.1,
          horizontal_flip=True,
          fill_mode='nearest',
          validation_split=0.2
      )

      val_datagen = ImageDataGenerator(
          rescale=1./255,
          validation_split=0.2
      )

      train_gen = train_datagen.flow_from_directory(
          dataset_path,
          target_size=img_size,
          batch_size=32,
          class_mode='categorical',
          subset='training',
          seed=SEED
      )

      val_gen = val_datagen.flow_from_directory(
          dataset_path,
          target_size=img_size,
          batch_size=32,
          class_mode='categorical',
          subset='validation',
          seed=SEED
      )

      return train_gen, val_gen
  ```
- **Penjelasan**:
  - **`import tensorflow as tf`**: Mengimpor TensorFlow, kerangka kerja utama untuk deep learning, yang menyediakan fungsi untuk membangun dan melatih model CNN.
  - **`from tensorflow.keras import layers, models, optimizers`**: Mengimpor modul spesifik dari Keras (bagian dari TensorFlow) untuk lapisan jaringan, model sequential, dan pengoptimal (optimizer).
  - **`from tensorflow.keras.preprocessing.image import ImageDataGenerator`**: Mengimpor utilitas untuk augmentasi dan pembuatan generator data gambar, memungkinkan preprocessing otomatis.
  - **`from tensorflow.keras.callbacks import EarlyStopping`**: Mengimpor callback untuk menghentikan pelatihan lebih awal jika performa tidak meningkat, mencegah overfit.
  - **`import numpy as np`**: Mengimpor NumPy untuk operasi matriks dan array, yang diperlukan untuk manipulasi data numerik.
  - **`import random`**: Mengimpor modul random untuk mengatur seed acak.
  - **`import os`**: Mengimpor modul os untuk operasi sistem file, meskipun tidak langsung digunakan di sini tetapi relevan untuk path data.
  - **`SEED = 42`**: Menetapkan nilai seed (42) untuk reprodusibilitas, sebuah konvensi umum dalam machine learning untuk memastikan hasil konsisten.
  - **`tf.random.set_seed(SEED)`**: Mengatur seed untuk TensorFlow’s random number generator, memengaruhi inisialisasi bobot dan operasi acak.
  - **`np.random.seed(SEED)`**: Mengatur seed untuk NumPy’s random number generator, memastikan konsistensi dalam operasi array.
  - **`random.seed(SEED)`**: Mengatur seed untuk modul random bawaan Python, menyelaraskan semua sumber acak.
  - **`def get_data_generators(img_size)`**: Mendefinisikan fungsi yang menerima parameter `img_size` (tuple, misalnya `(224, 224)`), mengembalikan generator data pelatihan dan validasi.
    - **`train_datagen = ImageDataGenerator(...)`**: Membuat generator untuk data pelatihan dengan:
      - `rescale=1./255`: Menormalkan intensitas piksel ke rentang [0, 1] untuk konsistensi.
      - `rotation_range=20`: Memutar gambar hingga 20 derajat secara acak untuk augmentasi.
      - `width_shift_range=0.1`: Menggeser lebar gambar hingga 10% secara acak.
      - `height_shift_range=0.1`: Menggeser tinggi gambar hingga 10% secara acak.
      - `shear_range=0.1`: Menerapkan shear transform hingga 10% untuk distorsi.
      - `zoom_range=0.1`: Memperbesar/memperkecil gambar hingga 10% secara acak.
      - `horizontal_flip=True`: Membalik gambar secara horizontal untuk variasi.
      - `fill_mode='nearest'`: Mengisi piksel kosong dengan nilai piksel terdekat setelah transformasi.
      - `validation_split=0.2`: Membagi 20% data untuk validasi.
    - **`val_datagen = ImageDataGenerator(...)`**: Generator untuk validasi, hanya dengan `rescale=1./255` dan `validation_split=0.2`, tanpa augmentasi untuk evaluasi murni.
    - **`train_gen = train_datagen.flow_from_directory(...)`**: Membuat generator pelatihan dari direktori `dataset_path` dengan:
      - `target_size=img_size`: Mengubah ukuran gambar ke `img_size`.
      - `batch_size=32`: Memproses 32 gambar sekaligus per iterasi.
      - `class_mode='categorical'`: Menggunakan one-hot encoding untuk label kelas.
      - `subset='training'`: Mengambil 80% data untuk pelatihan.
      - `seed=SEED`: Memastikan urutan data konsisten.
    - **`val_gen = val_datagen.flow_from_directory(...)`**: Sama seperti di atas, tetapi `subset='validation'` untuk 20% data validasi.
    - **`return train_gen, val_gen`**: Mengembalikan kedua generator untuk digunakan dalam pelatihan.

### Tahap 2: Pelatihan Model
  ```python
  def train_model(base_model_fn, model_name, input_shape):
      print(f"\n=== Training {model_name} ===")

      train_gen, val_gen = get_data_generators(input_shape)

      base_model = base_model_fn(weights='imagenet', include_top=False, input_shape=(*input_shape, 3))
      base_model.trainable = False

      model = models.Sequential([
          base_model,
          layers.GlobalAveragePooling2D(),
          layers.Dense(128, activation='relu'),
          layers.Dropout(0.5),
          layers.Dense(train_gen.num_classes, activation='softmax')
      ])

      model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

      early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

      # Train head
      model.fit(train_gen, validation_data=val_gen, epochs=3, callbacks=[early_stop])

      # Fine-tuning
      base_model.trainable = True
      for layer in base_model.layers[:-20]:
          layer.trainable = False

      model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

      model.fit(train_gen, validation_data=val_gen, epochs=40, callbacks=[early_stop])

      # Save model to Google Drive
      save_path = f'/content/drive/MyDrive/{model_name}.keras'
      model.save(save_path)
      print(f"Model saved to {save_path}")
  ```
- **Penjelasan**:
  - **`def train_model(base_model_fn, model_name, input_shape)`**: Fungsi yang menerima:
    - `base_model_fn`: Fungsi model pre-trained (misalnya, `tf.keras.applications.MobileNetV2`).
    - `model_name`: Nama model untuk penyimpanan (misalnya, "mobilenetv2").
    - `input_shape`: Tuple ukuran input gambar (misalnya, `(224, 224)`).
  - **`print(f"\n=== Training {model_name} ===")`**: Mencetak judul pelatihan untuk log.
  - **`train_gen, val_gen = get_data_generators(input_shape)`**: Memanggil fungsi sebelumnya untuk mendapatkan generator data.
  - **`base_model = base_model_fn(weights='imagenet', include_top=False, input_shape=(*input_shape, 3))`**:
    - `weights='imagenet'`: Memuat bobot pre-trained dari ImageNet.
    - `include_top=False`: Menghapus lapisan fully connected terakhir untuk transfer learning.
    - `input_shape=(*input_shape, 3)`: Menentukan dimensi input dengan 3 channel (RGB).
  - **`base_model.trainable = False`**: Membekukan lapisan pre-trained untuk melatih hanya lapisan baru.
  - **`model = models.Sequential([...])`**:
    - `layers.GlobalAveragePooling2D()`: Mengurangi dimensi fitur menjadi vektor rata-rata, menggantikan fully connected layer.
    - `layers.Dense(128, activation='relu')`: Lapisan dense dengan 128 neuron dan aktivasi ReLU untuk non-linearitas.
    - `layers.Dropout(0.5)`: Menghilangkan 50% neuron secara acak selama pelatihan untuk mencegah overfit.
    - `layers.Dense(train_gen.num_classes, activation='softmax')`: Lapisan output dengan jumlah kelas (6) dan aktivasi softmax untuk probabilitas kelas.
  - **`model.compile(...)`**:
    - `optimizer=optimizers.Adam(learning_rate=0.001)`: Pengoptimal Adam dengan learning rate 0.001 untuk pelatihan awal.
    - `loss='categorical_crossentropy'`: Fungsi loss untuk klasifikasi multi-kelas dengan one-hot encoding.
    - `metrics=['accuracy']`: Mengukur akurasi selama pelatihan.
  - **`early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`**:
    - `monitor='val_loss'`: Memantau loss validasi.
    - `patience=5`: Menunggu 5 epoch tanpa perbaikan sebelum berhenti.
    - `restore_best_weights=True`: Mengembalikan bobot terbaik berdasarkan `val_loss`.
  - **`model.fit(train_gen, validation_data=val_gen, epochs=3, callbacks=[early_stop])`**: Melatih "head" model selama 3 epoch dengan data pelatihan dan validasi.
  - **`base_model.trainable = True`**: Membuka semua lapisan untuk fine-tuning.
  - **`for layer in base_model.layers[:-20]: layer.trainable = False`**: Membekukan 20 lapisan terakhir, membiarkan sisanya dapat dilatih.
  - **`model.compile(...)`**: Mengkompilasi ulang dengan `learning_rate=1e-5` untuk fine-tuning yang lebih hati-hati.
  - **`model.fit(train_gen, validation_data=val_gen, epochs=40, callbacks=[early_stop])`**: Melatih ulang dengan maksimal 40 epoch.
  - **`save_path = f'/content/drive/MyDrive/{model_name}.keras'`**: Menentukan path penyimpanan di Google Drive.
  - **`model.save(save_path)`**: Menyimpan model dalam format Keras.
  - **`print(f"Model saved to {save_path}")`**: Konfirmasi penyimpanan.

### Tahap 3: Prediksi Ensemble
  ```python
  # Import required libraries
  import tensorflow as tf
  from tensorflow.keras.preprocessing import image
  import numpy as np
  from google.colab import files
  from google.colab import drive
  drive.mount('/content/drive')
  # Define class names (adjust if different in your dataset)
  class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

  # Load models
  model_1 = tf.keras.models.load_model('/content/drive/MyDrive/mobilenetv2.keras')
  model_2 = tf.keras.models.load_model('/content/drive/MyDrive/efficientnetb0.keras')
  model_3 = tf.keras.models.load_model('/content/drive/MyDrive/resnet50.keras')
  # Ensemble prediction functio
  def ensemble_predict(img_path, models):
      img = image.load_img(img_path, target_size=(224, 224))
      img_array = image.img_to_array(img) / 255.0
      img_array = np.expand_dims(img_array, axis=0)

      # Get predictions from all models
      predictions = [model.predict(img_array) for model in models]
      for i, model in enumerate(models):
          pred = model.predict(img_array)
          predicted_class = class_names[np.argmax(pred)]
          confidence = np.max(pred) * 100
          print(f"Model {i+1} Prediction: {predicted_class} ({confidence:.2f}%)")
          predictions.append(pred)

      avg_predictions = np.mean(predictions, axis=0)
      final_class = class_names[np.argmax(avg_predictions)]
      final_confidence = np.max(avg_predictions) * 100
      return final_class, final_confidence

  # Test ensemble
  uploaded = files.upload()
  for img_name in uploaded.keys():
      predicted_class, confidence = ensemble_predict(img_name, [model_1, model_2, model_3])
      print(f"Image: {img_name}, Predicted: {predicted_class}, Confidence: {confidence:.2f}%")
  ```
- **Penjelasan**:
  - **`import tensorflow as tf`**: Mengimpor TensorFlow untuk operasi model.
  - **`from tensorflow.keras.preprocessing import image`**: Mengimpor utilitas untuk memproses gambar.
  - **`import numpy as np`**: Untuk manipulasi array.
  - **`from google.colab import files`**: Mengimpor fungsi untuk mengunggah file di Colab.
  - **`from google.colab import drive`**: Mengimpor fungsi untuk mount Google Drive.
  - **`drive.mount('/content/drive')`**: Memasang Google Drive ke path `/content/drive` untuk akses file, meminta autentikasi pengguna.
  - **`class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']`**: Mendefinisikan daftar kelas sesuai dataset.
  - **`model_1 = tf.keras.models.load_model('/content/drive/MyDrive/mobilenetv2.keras')`**: Memuat model MobileNetV2 yang disimpan.
  - **`model_2 = tf.keras.models.load_model('/content/drive/MyDrive/efficientnetb0.keras')`**: Memuat model EfficientNetB0.
  - **`model_3 = tf.keras.models.load_model('/content/drive/MyDrive/resnet50.keras')`**: Memuat model ResNet50.
  - **`def ensemble_predict(img_path, models)`**: Fungsi yang menerima path gambar dan daftar model.
    - `img = image.load_img(img_path, target_size=(224, 224))`: Memuat dan mengubah ukuran gambar ke 224x224 piksel.
    - `img_array = image.img_to_array(img) / 255.0`: Mengonversi gambar ke array dan menormalkan ke [0, 1].
    - `img_array = np.expand_dims(img_array, axis=0)`: Menambahkan dimensi batch untuk kompatibilitas model.
    - `predictions = [model.predict(img_array) for model in models]`: Mendapatkan prediksi dari setiap model.
    - `for i, model in enumerate(models): ...`: Meloop melalui model, menghitung prediksi individu, dan mencetak hasil dengan `print`.
    - `avg_predictions = np.mean(predictions, axis=0)`: Menghitung rata-rata prediksi untuk ensemble.
    - `final_class = class_names[np.argmax(avg_predictions)]`: Menentukan kelas dengan probabilitas tertinggi.
    - `final_confidence = np.max(avg_predictions) * 100`: Mengonversi probabilitas maksimum ke persen.
    - `return final_class, final_confidence`: Mengembalikan kelas dan kepercayaan.
  - **`uploaded = files.upload()`**: Membuka dialog untuk mengunggah gambar.
  - **`for img_name in uploaded.keys(): ...`**: Meloop melalui gambar yang diunggah dan menampilkan hasil ensemble.

### Tahap 4: Evaluasi dengan Confusion Matrix dan Classification Report
  ```python
  # --- Evaluation: Confusion Matrix and Classification Report ---
  import os
  import seaborn as sns
  import matplotlib.pyplot as plt
  from sklearn.metrics import confusion_matrix, classification_report
  from tensorflow.keras.preprocessing.image import ImageDataGenerator
  from PIL import Image
  import tempfile

  dataset_path = '/content/drive/MyDrive/dataset-resized'
  SEED = 42
  val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
  val_generator = val_datagen.flow_from_directory(dataset_path, target_size=(224, 224), batch_size=1, class_mode='categorical', subset='validation', shuffle=False)

  predictions = []
  true_labels = []
  val_generator.reset()
  for i in range(len(val_generator)):
      img, label = next(val_generator)
      img_pil = Image.fromarray((img[0] * 255).astype(np.uint8))
      with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
          img_pil.save(tmp_file.name)
          # Suppress print output by redirecting stdout
          import sys
          original_stdout = sys.stdout
          sys.stdout = open(os.devnull, 'w')
          pred_class, _ = ensemble_predict(tmp_file.name, [model_1, model_2, model_3])
          sys.stdout = original_stdout
          predictions.append(class_names.index(pred_class))
      true_labels.append(np.argmax(label[0]))
      os.remove(tmp_file.name)

  cm = confusion_matrix(true_labels, predictions)
  plt.figure(figsize=(10, 8))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
  plt.title('Confusion Matrix for Ensemble Model')
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.savefig('/content/drive/MyDrive/confusion_matrix.png')
  plt.show()
  print("\nClassification Report:")
  print(classification_report(true_labels, predictions, target_names=class_names))
  ```
- **Penjelasan**:
  - **`import os`**: Mengimpor modul untuk operasi file.
  - **`import seaborn as sns`**: Mengimpor Seaborn untuk visualisasi heatmap.
  - **`import matplotlib.pyplot as plt`**: Mengimpor Matplotlib untuk plotting.
  - **`from sklearn.metrics import confusion_matrix, classification_report`**: Mengimpor metrik evaluasi dari Scikit-learn.
  - **`from tensorflow.keras.preprocessing.image import ImageDataGenerator`**: Untuk generator data gambar.
  - **`from PIL import Image`**: Mengimpor Pillow untuk memproses gambar.
  - **`import tempfile`**: Mengimpor utilitas untuk file sementara.
  - **`dataset_path = '/content/drive/MyDrive/dataset-resized'`**: Mendefinisikan path ke dataset yang sudah di-mount.
  - **`SEED = 42`**: Menetapkan seed untuk konsistensi.
  - **`val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)`**: Generator validasi dengan normalisasi dan pemisahan 20% data.
  - **`val_generator = val_datagen.flow_from_directory(...)`**: Membuat generator dengan:
    - `target_size=(224, 224)`: Mengubah ukuran gambar.
    - `batch_size=1`: Memproses satu gambar per iterasi.
    - `class_mode='categorical'`: Menggunakan one-hot encoding.
    - `subset='validation'`: Mengambil data validasi.
    - `shuffle=False`: Menjaga urutan label.
  - **`predictions = []`, `true_labels = []`**: Daftar untuk menyimpan hasil prediksi dan label sebenarnya.
  - **`val_generator.reset()`**: Mengatur ulang generator ke awal.
  - **`for i in range(len(val_generator)):`**: Meloop melalui 503 gambar validasi.
    - `img, label = next(val_generator)`: Mengambil batch gambar dan label.
    - `img_pil = Image.fromarray((img[0] * 255).astype(np.uint8))`: Mengonversi array gambar ke format PIL.
    - `with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:`: Membuat file sementara.
    - `img_pil.save(tmp_file.name)`: Menyimpan gambar sementara.
    - `import sys`, `original_stdout = sys.stdout`, `sys.stdout = open(os.devnull, 'w')`: Menekan output dengan mengarahkan stdout ke null.
    - `pred_class, _ = ensemble_predict(...)`: Mendapatkan prediksi tanpa print.
    - `sys.stdout = original_stdout`: Mengembalikan stdout normal.
    - `predictions.append(class_names.index(pred_class))`: Menyimpan indeks kelas prediksi.
    - `true_labels.append(np.argmax(label[0]))`: Menyimpan indeks label sebenarnya.
    - `os.remove(tmp_file.name)`: Menghapus file sementara.
  - **`cm = confusion_matrix(true_labels, predictions)`**: Menghitung matriks kebingungan.
  - **`plt.figure(figsize=(10, 8))`**: Membuat figure dengan ukuran 10x8 inci.
  - **`sns.heatmap(...)`**: Membuat heatmap dengan:
    - `annot=True`: Menampilkan nilai numerik.
    - `fmt='d'`: Format integer.
    - `cmap='Blues'`: Skema warna biru.
    - `xticklabels=class_names`, `yticklabels=class_names`: Label sumbu.
  - **`plt.title('Confusion Matrix for Ensemble Model')`**: Menambahkan judul.
  - **`plt.xlabel('Predicted')`, `plt.ylabel('True')`**: Menamakan sumbu.
  - **`plt.savefig('/content/drive/MyDrive/confusion_matrix.png')`**: Menyimpan heatmap.
  - **`plt.show()`**: Menampilkan heatmap.
  - **`print("\nClassification Report:")`, `print(classification_report(...))`**: Menampilkan laporan dengan nama kelas.
