import numpy as np
import time
import os
import json
import psutil
import matplotlib.pyplot as plt

# definisikan dimensi matriks yang akan diuji
dimensi = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
# dimensi_sangat_besar = [8192, 16384, 32768, 65536, 131072, 262144]

# buat folder untuk menyimpan hasil
os.makedirs("hasil_matriks", exist_ok=True)

# buat dictionary untuk menyimpan waktu eksekusi
waktu_numpy_dict = {}
waktu_perkalian_numpy_dict = {}

# Ambil informasi memori dari sistem
print("\n--- Informasi Sistem ---")
info_memori = psutil.virtual_memory()
print(f"Total RAM: {info_memori.total / (1024**3):.2f} GB")
print(f"RAM Tersedia: {info_memori.available / (1024**3):.2f} GB")
print(f"Persentase Terpakai: {info_memori.percent}%")

print("\n--- Memulai Pembuatan Matriks ---")

# fungsi untuk memeriksa apakah memori cukup tersedia
def periksa_memori_tersedia(dim):
    # Kita membutuhkan memori untuk dua matriks input dan satu matriks output
    memori_dibutuhkan = 3 * dim * dim * 8
    
    memori_tersedia = psutil.virtual_memory().available
    
    memori_aman = memori_tersedia * 0.95
    
    memori_cukup = memori_dibutuhkan <= memori_aman
    
    print(f"Matriks Ukuran: {dim}x{dim} Membutuhkan {memori_dibutuhkan / (1024**3):.2f} GB untuk perkalian")
    print(f"Memori Tersedia: {memori_tersedia / (1024**3):.2f} GB (menggunakan 95%: {memori_aman / (1024**3):.2f} GB)")
    print(f"Memori Cukup: {memori_cukup}")
    
    return memori_cukup

# Fungsi untuk membuat matriks numpy dan mengukur waktu
def buat_matriks_numpy(dim):
    # Periksa apakah memori cukup tersedia terlebih dahulu
    if not periksa_memori_tersedia(dim):
        print(f"Memori tidak cukup untuk matriks {dim}x{dim}. Melewati.")
        return None, 0
    
    try:
        waktu_mulai = time.time()
        matriks = np.random.randint(1, 101, size=(dim, dim))
        waktu_selesai = time.time()
        waktu_eksekusi = waktu_selesai - waktu_mulai
        print(f"NumPy {dim}x{dim} dibuat dalam {waktu_eksekusi:.6f} detik")
        return matriks, waktu_eksekusi
    except (MemoryError, np.core._exceptions._ArrayMemoryError) as e:
        print(f"Kesalahan memori saat membuat matriks {dim}x{dim}: {str(e)}")
        return None, 0
    except Exception as e:
        print(f"Kesalahan tak terduga saat membuat matriks {dim}x{dim}: {str(e)}")
        return None, 0

# Fungsi untuk melakukan perkalian matriks dan mengukur waktu
def perkalian_matriks_numpy(matriks_a, matriks_b):
    try:
        waktu_mulai = time.time()
        hasil = np.matmul(matriks_a, matriks_b)
        waktu_selesai = time.time()
        waktu_eksekusi = waktu_selesai - waktu_mulai
        return hasil, waktu_eksekusi
    except Exception as e:
        print(f"Kesalahan selama perkalian matriks: {str(e)}")
        return None, 0

# Lacak apakah ada kegagalan memori untuk menghentikan percobaan ukuran yang lebih besar
gagal_memori = False

# buat matriks untuk ukuran yang lebih kecil
for dim in dimensi:
    print(f"\nMembuat dan mengalikan matriks NumPy {dim}x{dim}...")
    
    # Buat matriks NumPy pertama
    matriks_numpy_a, waktu_buat_a = buat_matriks_numpy(dim)
    
    if matriks_numpy_a is not None:
        # Buat matriks NumPy kedua
        matriks_numpy_b, waktu_buat_b = buat_matriks_numpy(dim)
        
        if matriks_numpy_b is not None:
            waktu_numpy_dict[dim] = waktu_buat_a + waktu_buat_b
            
            # Lakukan perkalian matriks
            print(f"Mengalikan matriks {dim}x{dim}...")
            hasil, waktu_perkalian = perkalian_matriks_numpy(matriks_numpy_a, matriks_numpy_b)
            
            if hasil is not None:
                waktu_perkalian_numpy_dict[dim] = waktu_perkalian
                print(f"Perkalian matriks {dim}x{dim} selesai dalam {waktu_perkalian:.6f} detik")
                
                # Simpan contoh hasil matriks
                with open(f"hasil_matriks/hasil_perkalian_numpy_{dim}x{dim}_contoh.txt", "w") as f:
                    f.write(f"Hasil Perkalian NumPy {dim}x{dim} (menampilkan elemen 5x5 pertama):\n")
                    ukuran_contoh = min(5, dim)
                    for i in range(ukuran_contoh):
                        f.write(" ".join(map(str, hasil[i, :ukuran_contoh])) + "\n")
            
            # Bebaskan memori secara eksplisit
            del matriks_numpy_a, matriks_numpy_b
            if hasil is not None:
                del hasil
            import gc
            gc.collect()
            
            # Laporkan memori setelah setiap matriks besar
            if dim >= 1024:
                info_memori = psutil.virtual_memory()
                print(f"RAM Tersedia setelah dibersihkan: {info_memori.available / (1024**3):.2f} GB")
        else:
            gagal_memori = True
            break
    else:
        gagal_memori = True
        break

# Simpan hasil waktu ke file JSON
with open("hasil_matriks/waktu_pembuatan_numpy.json", "w") as f:
    json.dump(waktu_numpy_dict, f)

with open("hasil_matriks/waktu_perkalian_numpy.json", "w") as f:
    json.dump(waktu_perkalian_numpy_dict, f)

print("\nPembuatan dan perkalian matriks NumPy selesai.")
print(f"Hasil disimpan di hasil_matriks/waktu_pembuatan_numpy.json dan hasil_matriks/waktu_perkalian_numpy.json")

# Cetak ringkasan
print("\nRingkasan Pembuatan NumPy:")
print("=" * 50)
print(f"{'Dimensi':<10} {'Waktu (detik)':<15}")
print("-" * 50)
for dim in sorted(waktu_numpy_dict.keys()):
    print(f"{dim:<10} {waktu_numpy_dict[dim]:<15.6f}")

print("\nRingkasan Perkalian NumPy:")
print("=" * 50)
print(f"{'Dimensi':<10} {'Waktu (detik)':<15}")
print("-" * 50)
for dim in sorted(waktu_perkalian_numpy_dict.keys()):
    print(f"{dim:<10} {waktu_perkalian_numpy_dict[dim]:<15.6f}")

# Buat visualisasi menggunakan matplotlib
plt.figure(figsize=(12, 8))

# Plot waktu perkalian matriks
plt.subplot(2, 1, 1)
dimensi_sorted = sorted(waktu_perkalian_numpy_dict.keys())
waktu = [waktu_perkalian_numpy_dict[dim] for dim in dimensi_sorted]
plt.plot(dimensi_sorted, waktu, 'o-', color='blue', linewidth=2, markersize=8)
plt.title('Performa Perkalian Matriks NumPy', fontsize=14)
plt.xlabel('Dimensi Matriks', fontsize=12)
plt.ylabel('Waktu (detik)', fontsize=12)
plt.grid(True)
plt.xscale('log', base=2)
plt.yscale('log')

# Plot waktu pembuatan matriks
plt.subplot(2, 1, 2)
dimensi_sorted = sorted(waktu_numpy_dict.keys())
waktu = [waktu_numpy_dict[dim] for dim in dimensi_sorted]
plt.plot(dimensi_sorted, waktu, 'o-', color='green', linewidth=2, markersize=8)
plt.title('Performa Pembuatan Matriks NumPy', fontsize=14)
plt.xlabel('Dimensi Matriks', fontsize=12)
plt.ylabel('Waktu (detik)', fontsize=12)
plt.grid(True)
plt.xscale('log', base=2)
plt.yscale('log')

plt.tight_layout()
plt.savefig('hasil_matriks/performa_matriks_numpy.png')
print("Visualisasi performa disimpan di 'hasil_matriks/performa_matriks_numpy.png'")