import cupy as cp
import numpy as np
import time
import os
import json
import subprocess
import matplotlib.pyplot as plt

# Definisikan dimensi matriks yang akan diuji
dimensi_gpu = [8, 16, 32, 64, 128, 256, 512, 1024, 4096]

# Buat folder untuk menyimpan hasil
os.makedirs("matrix_results", exist_ok=True)

# Buat dictionary untuk menyimpan waktu eksekusi
waktu_pembuatan_cupy = {}
waktu_perkalian_cupy = {}

print("Memulai pembuatan matriks...")

# Jalankan nvidia-smi untuk mendapatkan informasi GPU
print("\n--- Informasi NVIDIA GPU ---")
try:
    output_nvidia_smi = subprocess.check_output(['nvidia-smi'], universal_newlines=True)
    print(output_nvidia_smi)
except (subprocess.SubprocessError, FileNotFoundError) as e:
    print(f"Error menjalankan Nvidia-smi: {e}")

# Ambil informasi memori dari GPU
total_memori_gpu = cp.cuda.runtime.memGetInfo()[1]
memori_gpu_tersedia = cp.cuda.runtime.memGetInfo()[0]
print(f"Total memori GPU (dilaporkan oleh CuPy): {total_memori_gpu / (1024**3):.2f} GB")
print(f"Memori GPU tersedia (dilaporkan oleh CuPy): {memori_gpu_tersedia / (1024**3):.2f} GB")
print("-------------------------------\n")

# Fungsi untuk membuat matriks CuPy dengan pemeriksaan keamanan memori
def buat_matriks_cupy(dim):
    # Hitung memori yang dibutuhkan dan periksa apakah cukup tersedia
    memori_dibutuhkan = dim * dim * 8  # 8 byte per float64
    memori_tersedia = cp.cuda.runtime.memGetInfo()[0]  # Ambil memori bebas
    
    if memori_dibutuhkan > memori_tersedia * 0.95:  # Gunakan hanya 95% memori sebagai keamanan
        print(f"Peringatan: Memori GPU tidak cukup untuk matriks {dim}x{dim}.")
        print(f"Dibutuhkan: {memori_dibutuhkan / (1024**3):.2f} GB, Tersedia: {memori_tersedia / (1024**3):.2f} GB")
        return None, 0
    
    try:
        waktu_mulai = time.time()
        matriks = cp.random.randint(1, 101, size=(dim, dim))
        # Sinkronisasi untuk mendapatkan waktu yang akurat
        cp.cuda.Stream.null.synchronize()
        waktu_selesai = time.time()
        waktu_eksekusi = waktu_selesai - waktu_mulai
        print(f"Matriks CuPy {dim}x{dim} dibuat dalam {waktu_eksekusi:.6f} detik")
        return matriks, waktu_eksekusi
    except cp.cuda.memory.OutOfMemoryError:
        print(f"Kehabisan memori untuk matriks {dim}x{dim}.")
        return None, 0

# Fungsi untuk melakukan perkalian matriks menggunakan CuPy
def perkalian_matriks_cupy(matriks_a, matriks_b):
    try:
        # Sinkronisasi perangkat sebelum memulai pengukuran waktu
        cp.cuda.Stream.null.synchronize()
        waktu_mulai = time.time()
        
        hasil = cp.matmul(matriks_a, matriks_b)
        
        # Pastikan operasi selesai sebelum menghentikan pengukuran waktu
        cp.cuda.Stream.null.synchronize()
        waktu_selesai = time.time()
        
        waktu_eksekusi = waktu_selesai - waktu_mulai
        print(f"Perkalian matriks selesai dalam {waktu_eksekusi:.6f} detik")
        return hasil, waktu_eksekusi
    except Exception as e:
        print(f"Error selama perkalian matriks CuPy: {str(e)}")
        return None, 0

# Membuat matriks CuPy dan melakukan perkalian
for dim in dimensi_gpu:
    print(f"\nPembuatan dan perkalian matriks CuPy {dim}x{dim}...")
    
    # Periksa apakah memori cukup untuk dua matriks dan hasilnya
    memori_dibutuhkan = 3 * dim * dim * 8  # Tiga matriks: A, B, dan hasil
    memori_tersedia = cp.cuda.runtime.memGetInfo()[0]
    
    if memori_dibutuhkan > memori_tersedia * 0.95:
        print(f"Memori tidak cukup untuk perkalian matriks {dim}x{dim}.")
        print(f"Dibutuhkan: {memori_dibutuhkan / (1024**3):.2f} GB, Tersedia: {memori_tersedia / (1024**3):.2f} GB")
        continue
    
    # Buat matriks CuPy pertama
    matriks_cupy_a, waktu_gen_a = buat_matriks_cupy(dim)
    
    if matriks_cupy_a is not None:
        # Buat matriks CuPy kedua
        matriks_cupy_b, waktu_gen_b = buat_matriks_cupy(dim)
        
        if matriks_cupy_b is not None:
            waktu_pembuatan_cupy[dim] = waktu_gen_a + waktu_gen_b
            
            # Lakukan perkalian matriks
            print(f"Melakukan perkalian matriks {dim}x{dim}...")
            hasil, waktu_mult = perkalian_matriks_cupy(matriks_cupy_a, matriks_cupy_b)
            
            if hasil is not None:
                waktu_perkalian_cupy[dim] = waktu_mult
                
                # Simpan sampel hasil matriks
                with open(f"matrix_results/hasil_perkalian_cupy_{dim}x{dim}_sampel.txt", "w") as f:
                    f.write(f"Hasil Perkalian CuPy {dim}x{dim} (menampilkan elemen 5x5 pertama):\n")
                    ukuran_sampel = min(5, dim)
                    hasil_np = cp.asnumpy(hasil[:ukuran_sampel, :ukuran_sampel])  # Konversi bagian kecil ke NumPy
                    for i in range(ukuran_sampel):
                        f.write(" ".join(map(str, hasil_np[i])) + "\n")
            
            # Bebaskan memori secara eksplisit
            del matriks_cupy_a, matriks_cupy_b
            if hasil is not None:
                del hasil
            cp.get_default_memory_pool().free_all_blocks()
            print(f"Memori GPU yang tersedia: {cp.cuda.runtime.memGetInfo()[0] / (1024**3):.2f} GB")

# Simpan hasil waktu eksekusi ke dalam file JSON
with open("matrix_results/waktu_pembuatan_cupy.json", "w") as f:
    json.dump(waktu_pembuatan_cupy, f)

with open("matrix_results/waktu_perkalian_cupy.json", "w") as f:
    json.dump(waktu_perkalian_cupy, f)

print("\nPembuatan dan perkalian matriks selesai.")
print(f"Hasil disimpan ke matrix_results/waktu_pembuatan_cupy.json dan matrix_results/waktu_perkalian_cupy.json")

# Cetak kesimpulan - waktu pembuatan
print("\nKesimpulan Waktu Pembuatan CuPy:")
print("=" * 50)
print(f"{'Dimensi':<10} {'Waktu (Detik)':<15}")
print("-" * 50)
for dim in sorted(waktu_pembuatan_cupy.keys()):
    print(f"{dim:<10} {waktu_pembuatan_cupy[dim]:<15.6f}")

# Cetak kesimpulan - waktu perkalian
print("\nKesimpulan Waktu Perkalian CuPy:")
print("=" * 50)
print(f"{'Dimensi':<10} {'Waktu (Detik)':<15}")
print("-" * 50)
for dim in sorted(waktu_perkalian_cupy.keys()):
    print(f"{dim:<10} {waktu_perkalian_cupy[dim]:<15.6f}")

# Buat visualisasi menggunakan matplotlib
plt.figure(figsize=(12, 8))

# Plot waktu perkalian matriks
plt.subplot(2, 1, 1)
dimensi = sorted(waktu_perkalian_cupy.keys())
waktu = [waktu_perkalian_cupy[dim] for dimensi in dimensi]
plt.plot(dimensi, waktu, 'o-', color='red', linewidth=2, markersize=8)
plt.title('Performa Perkalian Matriks CuPy (CUDA)', fontsize=14)
plt.xlabel('Dimensi Matriks', fontsize=12)
plt.ylabel('Waktu (detik)', fontsize=12)
plt.grid(True)
plt.xscale('log', base=2)
plt.yscale('log')

# Plot waktu pembuatan matriks
plt.subplot(2, 1, 2)
dimensi = sorted(waktu_pembuatan_cupy.keys())
waktu = [waktu_pembuatan_cupy[dim] for dimensi in dimensi]
plt.plot(dimensi, waktu, 'o-', color='orange', linewidth=2, markersize=8)
plt.title('Performa Pembuatan Matriks CuPy', fontsize=14)
plt.xlabel('Dimensi Matriks', fontsize=12)
plt.ylabel('Waktu (detik)', fontsize=12)
plt.grid(True)
plt.xscale('log', base=2)
plt.yscale('log')

plt.tight_layout()
plt.savefig('matrix_results/performa_matriks_cupy.png')
print("Visualisasi performa disimpan ke 'matrix_results/performa_matriks_cupy.png'")