import matplotlib.pyplot as plt
import json
import os
import numpy as np

# Buat direktori untuk plot jika belum ada
os.makedirs("matrix_results", exist_ok=True)

# Muat data waktu untuk pembuatan matriks
try:
    with open("matrix_results/numpy_generation_times.json", "r") as f:
        numpy_gen_times_dict = json.load(f)
    print("Data waktu pembuatan NumPy berhasil dimuat")
except FileNotFoundError:
    try:
        with open("matrix_results/waktu_pembuatan_numpy.json", "r") as f:
            numpy_gen_times_dict = json.load(f)
        print("Data waktu pembuatan NumPy berhasil dimuat (jalur alternatif)")
    except FileNotFoundError:
        print("Data waktu pembuatan NumPy tidak ditemukan")
        numpy_gen_times_dict = {}

try:
    with open("matrix_results/cupy_generation_times.json", "r") as f:
        cupy_gen_times_dict = json.load(f)
    print("Data waktu pembuatan CuPy berhasil dimuat")
except FileNotFoundError:
    try:
        with open("matrix_results/waktu_pembuatan_cupy.json", "r") as f:
            cupy_gen_times_dict = json.load(f)
        print("Data waktu pembuatan CuPy berhasil dimuat (jalur alternatif)")
    except FileNotFoundError:
        print("Data waktu pembuatan CuPy tidak ditemukan")
        cupy_gen_times_dict = {}

# Muat data waktu untuk perkalian matriks
try:
    with open("matrix_results/numpy_multiplication_times.json", "r") as f:
        numpy_mult_times_dict = json.load(f)
    print("Data waktu perkalian NumPy berhasil dimuat")
except FileNotFoundError:
    try:
        with open("matrix_results/waktu_perkalian_numpy.json", "r") as f:
            numpy_mult_times_dict = json.load(f)
        print("Data waktu perkalian NumPy berhasil dimuat (jalur alternatif)")
    except FileNotFoundError:
        print("Data waktu perkalian NumPy tidak ditemukan")
        numpy_mult_times_dict = {}

try:
    with open("matrix_results/cupy_multiplication_times.json", "r") as f:
        cupy_mult_times_dict = json.load(f)
    print("Data waktu perkalian CuPy berhasil dimuat")
except FileNotFoundError:
    try:
        with open("matrix_results/waktu_perkalian_cupy.json", "r") as f:
            cupy_mult_times_dict = json.load(f)
        print("Data waktu perkalian CuPy berhasil dimuat (jalur alternatif)")
    except FileNotFoundError:
        print("Data waktu perkalian CuPy tidak ditemukan")
        cupy_mult_times_dict = {}

# Ubah kunci string menjadi integer (JSON mengubah kunci menjadi string)
numpy_gen_times_dict = {int(k): v for k, v in numpy_gen_times_dict.items()}
cupy_gen_times_dict = {int(k): v for k, v in cupy_gen_times_dict.items()}
numpy_mult_times_dict = {int(k): v for k, v in numpy_mult_times_dict.items()}
cupy_mult_times_dict = {int(k): v for k, v in cupy_mult_times_dict.items()}

# Fungsi visualisasi untuk menghindari duplikasi kode
def buat_plot_perbandingan(waktu_numpy, waktu_cupy, judul, nama_file, tipe_operasi):
    if not waktu_numpy or not waktu_cupy:
        print(f"Data tidak cukup untuk membuat plot perbandingan {tipe_operasi}.")
        return False
        
    plt.figure(figsize=(12, 7))
    
    # Plot waktu NumPy
    np_dims = sorted(waktu_numpy.keys())
    np_times = [waktu_numpy[d] for d in np_dims]
    plt.plot(np_dims, np_times, 'o-', label='NumPy', color='blue', linewidth=2, markersize=8)
    
    # Plot waktu CuPy
    cp_dims = sorted(waktu_cupy.keys())
    cp_times = [waktu_cupy[d] for d in cp_dims]
    plt.plot(cp_dims, cp_times, 's-', label='CuPy (CUDA)', color='green', linewidth=2, markersize=8)
    
    plt.xlabel('Dimensi Matriks', fontsize=12)
    plt.ylabel('Waktu Eksekusi (detik)', fontsize=12)
    plt.title(judul, fontsize=14)
    plt.legend(fontsize=12)
    
    # Tambahkan grid untuk keterbacaan yang lebih baik
    plt.grid(True)
    
    # Buat ticks yang lebih rapi (tidak terlalu padat)
    semua_dimensi = sorted(list(set(list(waktu_numpy.keys()) + list(waktu_cupy.keys()))))
    
    # Jika jumlah dimensi terlalu banyak, pilih subset untuk ditampilkan
    if len(semua_dimensi) > 10:
        # Pilih maksimal 10 dimensi yang terdistribusi merata
        step = max(1, len(semua_dimensi) // 10)
        tick_dimensions = semua_dimensi[::step]
        
        # Pastikan dimensi terbesar selalu ditampilkan
        if semua_dimensi[-1] not in tick_dimensions:
            tick_dimensions.append(semua_dimensi[-1])
            
        plt.xticks(tick_dimensions, [str(d) for d in tick_dimensions], rotation=45)
    else:
        plt.xticks(semua_dimensi, [str(d) for d in semua_dimensi], rotation=45)
    
    # Buat skala y-axis yang lebih rapi
    y_data = np_times + cp_times
    y_min, y_max = min(y_data), max(y_data)
    
    # Atur ticks pada sumbu y agar lebih terdistribusi merata
    y_range = y_max - y_min
    if y_range > 0:
        num_ticks = min(10, max(5, len(y_data)))  # Antara 5-10 ticks
        plt.yticks(np.linspace(0, y_max * 1.05, num_ticks))
    
    plt.tight_layout()
    plt.savefig(nama_file)
    print(f"Plot perbandingan {tipe_operasi} disimpan ke '{nama_file}'")
    
    # Hitung dan cetak percepatan
    print(f"\nRingkasan Percepatan {tipe_operasi.capitalize()}:")
    print("=" * 70)
    print(f"{'Dimensi':<10} {'Waktu NumPy (s)':<15} {'Waktu CuPy (s)':<15} {'Percepatan':<10}")
    print("-" * 70)
    
    dimensi_umum = sorted(set(waktu_numpy.keys()) & set(waktu_cupy.keys()))
    
    percepatan = []
    for dim in dimensi_umum:
        np_time = waktu_numpy.get(dim, float('nan'))
        cp_time = waktu_cupy.get(dim, float('nan'))
        
        if np_time > 0 and cp_time > 0:
            speedup = np_time / cp_time
            percepatan.append(speedup)
            print(f"{dim:<10} {np_time:<15.6f} {cp_time:<15.6f} {speedup:<10.2f}x")
        else:
            print(f"{dim:<10} {np_time:<15.6f} {cp_time:<15.6f} {'N/A':<10}")
    
    if percepatan:
        rata_rata_percepatan = np.mean(percepatan)
        print(f"Rata-rata Percepatan: {rata_rata_percepatan:.2f}x")
    
    return True

# Buat visualisasi untuk perbandingan pembuatan matriks
plot_pembuatan_dibuat = buat_plot_perbandingan(
    numpy_gen_times_dict,
    cupy_gen_times_dict,
    'Performa Pembuatan Matriks: NumPy vs CuPy (CUDA)',
    'matrix_results/generation_comparison.png',
    'pembuatan'
)

# Buat visualisasi untuk perbandingan perkalian matriks
plot_perkalian_dibuat = buat_plot_perbandingan(
    numpy_mult_times_dict,
    cupy_mult_times_dict,
    'Performa Perkalian Matriks: NumPy vs CuPy (CUDA)',
    'matrix_results/multiplication_comparison.png',
    'perkalian'
)

# Buat visualisasi gabungan jika kedua operasi memiliki data
if plot_pembuatan_dibuat and plot_perkalian_dibuat:
    plt.figure(figsize=(15, 10))
    
    # Subplot perbandingan pembuatan
    plt.subplot(2, 1, 1)
    np_dims = sorted(numpy_gen_times_dict.keys())
    np_times = [numpy_gen_times_dict[d] for d in np_dims]
    plt.plot(np_dims, np_times, 'o-', label='NumPy', color='blue', linewidth=2)
    
    cp_dims = sorted(cupy_gen_times_dict.keys())
    cp_times = [cupy_gen_times_dict[d] for d in cp_dims]
    plt.plot(cp_dims, cp_times, 's-', label='CuPy (CUDA)', color='green', linewidth=2)
    
    plt.xlabel('Dimensi Matriks')
    plt.ylabel('Waktu (detik)')
    plt.title('Perbandingan Performa Pembuatan Matriks')
    plt.legend()
    plt.grid(True)
    
    # Buat ticks yang lebih rapi
    semua_dimensi = sorted(list(set(list(numpy_gen_times_dict.keys()) + list(cupy_gen_times_dict.keys()))))
    if len(semua_dimensi) > 10:
        step = max(1, len(semua_dimensi) // 8)
        tick_dimensions = semua_dimensi[::step]
        if semua_dimensi[-1] not in tick_dimensions:
            tick_dimensions.append(semua_dimensi[-1])
        plt.xticks(tick_dimensions, [str(d) for d in tick_dimensions], rotation=45)
    
    # Subplot perbandingan perkalian
    plt.subplot(2, 1, 2)
    np_dims = sorted(numpy_mult_times_dict.keys())
    np_times = [numpy_mult_times_dict[d] for d in np_dims]
    plt.plot(np_dims, np_times, 'o-', label='NumPy', color='blue', linewidth=2)
    
    cp_dims = sorted(cupy_mult_times_dict.keys())
    cp_times = [cupy_mult_times_dict[d] for d in cp_dims]
    plt.plot(cp_dims, cp_times, 's-', label='CuPy (CUDA)', color='green', linewidth=2)
    
    plt.xlabel('Dimensi Matriks')
    plt.ylabel('Waktu (detik)')
    plt.title('Perbandingan Performa Perkalian Matriks')
    plt.legend()
    plt.grid(True)
    
    # Buat ticks yang lebih rapi
    semua_dimensi = sorted(list(set(list(numpy_mult_times_dict.keys()) + list(cupy_mult_times_dict.keys()))))
    if len(semua_dimensi) > 10:
        step = max(1, len(semua_dimensi) // 8)
        tick_dimensions = semua_dimensi[::step]
        if semua_dimensi[-1] not in tick_dimensions:
            tick_dimensions.append(semua_dimensi[-1])
        plt.xticks(tick_dimensions, [str(d) for d in tick_dimensions], rotation=45)
    
    plt.tight_layout()
    plt.savefig('matrix_results/combined_performance_comparison.png')
    print("Perbandingan performa gabungan disimpan ke 'matrix_results/combined_performance_comparison.png'")

if not plot_pembuatan_dibuat and not plot_perkalian_dibuat:
    print("Data tidak cukup untuk membuat plot perbandingan. Pastikan skrip NumPy dan CuPy telah dijalankan.")

plt.show()