#include <immintrin.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <omp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstring>

namespace solution {

class SparseAccumulator {
private:
    static constexpr size_t INIT_SIZE = 1024;
    static constexpr size_t LOAD_FACTOR_THRESHOLD = 70; 
    static constexpr double RESIZE_FACTOR = 2.0;
    
    int* indices;          
    double* values;        
    bool* occupied;        
    size_t capacity;       
    size_t size;           
    size_t table_size;     
    size_t mask;           
    
public:
    SparseAccumulator(int expected_nnz = INIT_SIZE) {
        
        table_size = 1;
        while (table_size < expected_nnz * 2) table_size <<= 1;
        
        capacity = expected_nnz < table_size/2 ? expected_nnz : table_size/2;
        mask = table_size - 1;
        size = 0;
        
        indices = static_cast<int*>(aligned_alloc(64, table_size * sizeof(int)));
        values = static_cast<double*>(aligned_alloc(64, table_size * sizeof(double)));
        occupied = static_cast<bool*>(aligned_alloc(64, table_size * sizeof(bool)));
        
        std::memset(occupied, 0, table_size * sizeof(bool));
    }
    
    ~SparseAccumulator() {
        free(indices);
        free(values);
        free(occupied);
    }
    
    SparseAccumulator(const SparseAccumulator&) = delete;
    SparseAccumulator& operator=(const SparseAccumulator&) = delete;
    
    inline void add(int col, double val) {
        
        if (size >= capacity) {
            resize();
        }
        
        size_t pos = col & mask;
        while (true) {
            if (!occupied[pos]) {
                
                indices[pos] = col;
                values[pos] = val;
                occupied[pos] = true;
                size++;
                return;
            } else if (indices[pos] == col) {
                
                values[pos] += val;
                return;
            }
            
            pos = (pos + 1) & mask;
        }
    }
    
    std::vector<std::pair<int, double>> get_sorted_entries(double threshold = 1e-14) {
        std::vector<std::pair<int, double>> result;
        result.reserve(size);
        
        for (size_t i = 0; i < table_size; i++) {
            if (occupied[i] && std::abs(values[i]) >= threshold) {
                result.emplace_back(indices[i], values[i]);
            }
        }
        
        std::sort(result.begin(), result.end());
        return result;
    }
    
    inline void clear() {
        std::memset(occupied, 0, table_size * sizeof(bool));
        size = 0;
    }
    
private:
    void resize() {
        size_t old_table_size = table_size;
        bool* old_occupied = occupied;
        int* old_indices = indices;
        double* old_values = values;
        
        table_size *= RESIZE_FACTOR;
        capacity = table_size * LOAD_FACTOR_THRESHOLD / 100;
        mask = table_size - 1;
        
        indices = static_cast<int*>(aligned_alloc(64, table_size * sizeof(int)));
        values = static_cast<double*>(aligned_alloc(64, table_size * sizeof(double)));
        occupied = static_cast<bool*>(aligned_alloc(64, table_size * sizeof(bool)));
        
        std::memset(occupied, 0, table_size * sizeof(bool));
        
        size = 0;
        for (size_t i = 0; i < old_table_size; i++) {
            if (old_occupied[i]) {
                add(old_indices[i], old_values[i]);
            }
        }
        
        free(old_occupied);
        free(old_indices);
        free(old_values);
    }
};

inline int estimate_row_nnz(const int* A_row_ptr, int row, 
                           const double* A_values, const int* A_col_ind,
                           const int* B_row_ptr, int k, int n) {
    int A_nnz = A_row_ptr[row + 1] - A_row_ptr[row];
    if (A_nnz == 0) return 0;
    
    double A_density = static_cast<double>(A_nnz) / k;
    double B_total_nnz = static_cast<double>(B_row_ptr[k]);
    double B_avg_density = B_total_nnz / (k * n);
    
    double expected_density = A_density * B_avg_density * 1.5 * k;
    expected_density = std::min(expected_density, 0.25); 
    
    return static_cast<int>(expected_density * n) + 4; 
}

void sparse_spmm(
    const double* A_values, const int* A_col_ind, const int* A_row_ptr,
    const double* B_values, const int* B_col_ind, const int* B_row_ptr,
    double** C_values, int** C_col_ind, int** C_row_ptr,
    int m, int k, int n
) {
    
    *C_row_ptr = static_cast<int*>(aligned_alloc(64, (m + 1) * sizeof(int)));
    (*C_row_ptr)[0] = 0;
    
    std::vector<int> row_nnz_estimate(m);
    int est_total_nnz = 0;
    
    #pragma omp parallel for reduction(+:est_total_nnz) schedule(static)
    for (int i = 0; i < m; i++) {
        row_nnz_estimate[i] = estimate_row_nnz(A_row_ptr, i, A_values, A_col_ind, B_row_ptr, k, n);
        est_total_nnz += row_nnz_estimate[i];
    }
    
    std::vector<std::vector<std::pair<int, double>>> C_temp(m);
    
    int chunk_size = std::max(1, std::min(64, m / (omp_get_max_threads() * 2)));
    
    #pragma omp parallel
    {
        
        SparseAccumulator spa(n/8 + 16); 
        
        #pragma omp for schedule(dynamic, chunk_size)
        for (int i = 0; i < m; i++) {
            const int row_nnz = A_row_ptr[i+1] - A_row_ptr[i];
            
            if (row_nnz == 0) {
                C_temp[i].clear();
                continue;
            }
            
            spa.clear();
            
            for (int jA = A_row_ptr[i]; jA < A_row_ptr[i+1]; jA++) {
                const int col_A = A_col_ind[jA];
                const double val_A = A_values[jA];
                
                if (std::abs(val_A) < 1e-14) continue;
                
                if (jA + 1 < A_row_ptr[i+1]) {
                    __builtin_prefetch(&A_col_ind[jA + 1], 0, 3);
                    __builtin_prefetch(&A_values[jA + 1], 0, 3);
                }
                
                __builtin_prefetch(&B_row_ptr[col_A], 0, 3);
                
                const int B_row_start = B_row_ptr[col_A];
                const int B_row_end = B_row_ptr[col_A + 1];
                
                if (B_row_start < B_row_end) {
                    __builtin_prefetch(&B_col_ind[B_row_start], 0, 3);
                    __builtin_prefetch(&B_values[B_row_start], 0, 3);
                }
                
                for (int jB = B_row_start; jB < B_row_end; jB++) {
                    
                    if (jB + 1 < B_row_end) {
                        __builtin_prefetch(&B_col_ind[jB + 1], 0, 3);
                        __builtin_prefetch(&B_values[jB + 1], 0, 3);
                    }
                    
                    const int col_B = B_col_ind[jB];
                    const double val_B = B_values[jB];
                    const double product = val_A * val_B;
                    
                    if (std::abs(product) >= 1e-14) {
                        spa.add(col_B, product);
                    }
                }
            }
            
            C_temp[i] = spa.get_sorted_entries(1e-14);
        }
    }
    
    (*C_row_ptr)[0] = 0;
    for (int i = 0; i < m; i++) {
        (*C_row_ptr)[i + 1] = (*C_row_ptr)[i] + C_temp[i].size();
    }
    
    const int nnz = (*C_row_ptr)[m];
    *C_values = static_cast<double*>(aligned_alloc(64, nnz * sizeof(double)));
    *C_col_ind = static_cast<int*>(aligned_alloc(64, nnz * sizeof(int)));
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        const int row_start = (*C_row_ptr)[i];
        const int row_size = C_temp[i].size();
        
        for (int j = 0; j < row_size; j++) {
            (*C_col_ind)[row_start + j] = C_temp[i][j].first;
            (*C_values)[row_start + j] = C_temp[i][j].second;
        }
        
        std::vector<std::pair<int, double>>().swap(C_temp[i]);
    }
}

#ifdef __AVX512F__
inline void micro_kernel_4x16(const float* A, const float* B, float* C, 
                             int k, int lda, int ldb, int ldc) {
    __m512 c0 = _mm512_loadu_ps(&C[0*ldc]);
    __m512 c1 = _mm512_loadu_ps(&C[1*ldc]);
    __m512 c2 = _mm512_loadu_ps(&C[2*ldc]);
    __m512 c3 = _mm512_loadu_ps(&C[3*ldc]);
    
    for (int l = 0; l < k; ++l) {
        __m512 b = _mm512_loadu_ps(&B[l*ldb]);
        
        __m512 a0 = _mm512_set1_ps(A[0*lda + l]);
        __m512 a1 = _mm512_set1_ps(A[1*lda + l]);
        __m512 a2 = _mm512_set1_ps(A[2*lda + l]);
        __m512 a3 = _mm512_set1_ps(A[3*lda + l]);
        
        c0 = _mm512_fmadd_ps(a0, b, c0);
        c1 = _mm512_fmadd_ps(a1, b, c1);
        c2 = _mm512_fmadd_ps(a2, b, c2);
        c3 = _mm512_fmadd_ps(a3, b, c3);
    }
    
    _mm512_storeu_ps(&C[0*ldc], c0);
    _mm512_storeu_ps(&C[1*ldc], c1);
    _mm512_storeu_ps(&C[2*ldc], c2);
    _mm512_storeu_ps(&C[3*ldc], c3);
}
#elif defined(__AVX2__)
inline void micro_kernel_4x16(const float* A, const float* B, float* C, 
                             int k, int lda, int ldb, int ldc) {
    __m256 c00 = _mm256_loadu_ps(&C[0*ldc]);
    __m256 c01 = _mm256_loadu_ps(&C[0*ldc + 8]);
    __m256 c10 = _mm256_loadu_ps(&C[1*ldc]);
    __m256 c11 = _mm256_loadu_ps(&C[1*ldc + 8]);
    __m256 c20 = _mm256_loadu_ps(&C[2*ldc]);
    __m256 c21 = _mm256_loadu_ps(&C[2*ldc + 8]);
    __m256 c30 = _mm256_loadu_ps(&C[3*ldc]);
    __m256 c31 = _mm256_loadu_ps(&C[3*ldc + 8]);
    
    for (int l = 0; l < k; ++l) {
        __m256 b0 = _mm256_loadu_ps(&B[l*ldb]);
        __m256 b1 = _mm256_loadu_ps(&B[l*ldb + 8]);
        
        __m256 a0 = _mm256_set1_ps(A[0*lda + l]);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        
        __m256 a1 = _mm256_set1_ps(A[1*lda + l]);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        
        __m256 a2 = _mm256_set1_ps(A[2*lda + l]);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        
        __m256 a3 = _mm256_set1_ps(A[3*lda + l]);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
    }
    
    _mm256_storeu_ps(&C[0*ldc], c00);
    _mm256_storeu_ps(&C[0*ldc + 8], c01);
    _mm256_storeu_ps(&C[1*ldc], c10);
    _mm256_storeu_ps(&C[1*ldc + 8], c11);
    _mm256_storeu_ps(&C[2*ldc], c20);
    _mm256_storeu_ps(&C[2*ldc + 8], c21);
    _mm256_storeu_ps(&C[3*ldc], c30);
    _mm256_storeu_ps(&C[3*ldc + 8], c31);
}
#else
inline void micro_kernel_4x16(const float* A, const float* B, float* C, 
                             int k, int lda, int ldb, int ldc) {
    float c[4][16] = {0};
    
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 16; j++) {
            c[i][j] = C[i*ldc + j];
        }
    }
    
    for (int l = 0; l < k; l++) {
        for (int i = 0; i < 4; i++) {
            float a = A[i*lda + l];
            #pragma unroll 16
            for (int j = 0; j < 16; j++) {
                c[i][j] += a * B[l*ldb + j];
            }
        }
    }
    
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 16; j++) {
            C[i*ldc + j] = c[i][j];
        }
    }
}
#endif

inline void block_multiply(const float* A, const float* B, float* C, 
                          int mb, int nb, int kb, int lda, int ldb, int ldc) {
    constexpr int MR = 4;   
    constexpr int NR = 16;  
    
    for (int i = 0; i < mb; i += MR) {
        const int ib = std::min(i + MR, mb);
        
        for (int j = 0; j < nb; j += NR) {
            const int jb = std::min(j + NR, nb);
            
            if (ib - i == MR && jb - j == NR) {
                micro_kernel_4x16(&A[i*lda], &B[j], &C[i*ldc + j], kb, lda, ldb, ldc);
            } else {
                for (int ii = i; ii < ib; ++ii) {
                    for (int jj = j; jj < jb; ++jj) {
                        float sum = C[ii*ldc + jj];
                        for (int l = 0; l < kb; ++l) {
                            sum += A[ii*lda + l] * B[l*ldb + jj];
                        }
                        C[ii*ldc + jj] = sum;
                    }
                }
            }
        }
    }
}

std::string compute(const std::string &m1_path, const std::string &m2_path, int n, int k, int m) {
    std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.dat";
    std::ofstream sol_fs(sol_path, std::ios::binary);
    std::ifstream m1_fs(m1_path, std::ios::binary), m2_fs(m2_path, std::ios::binary);
    
    const auto m1 = std::unique_ptr<float[], decltype(&free)>((float*)aligned_alloc(64, n*k*sizeof(float)), free);
    const auto m2 = std::unique_ptr<float[], decltype(&free)>((float*)aligned_alloc(64, k*m*sizeof(float)), free);
    
    m1_fs.read(reinterpret_cast<char*>(m1.get()), sizeof(float) * n * k);
    m2_fs.read(reinterpret_cast<char*>(m2.get()), sizeof(float) * k * m);
    m1_fs.close();
    m2_fs.close();
    
    auto result = std::unique_ptr<float[], decltype(&free)>((float*)aligned_alloc(64, n*m*sizeof(float)), free);
    std::memset(result.get(), 0, n*m*sizeof(float));
    
    const int MC = 32;     
    const int KC = 64;    
    const int NC = 128;    
    
    int max_threads = omp_get_max_threads();
    int num_threads = std::min(max_threads, std::max(1, (n + MC - 1) / MC));
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel
    {
        
        alignas(64) float A_tile[MC][KC];
        alignas(64) float B_tile[KC][NC];
        alignas(64) float C_tile[MC][NC];
        
        #pragma omp for schedule(dynamic, 1)
        for (int i0 = 0; i0 < n; i0 += MC) {
            const int mb = std::min(MC, n - i0);
            
            for (int j0 = 0; j0 < m; j0 += NC) {
                const int nb = std::min(NC, m - j0);
                
                for (int i = 0; i < mb; i++) {
                    for (int j = 0; j < nb; j++) {
                        C_tile[i][j] = 0.0f;
                    }
                }
                
                for (int k0 = 0; k0 < k; k0 += KC) {
                    const int kb = std::min(KC, k - k0);
                    
                    for (int i = 0; i < mb; i++) {
                        for (int kk = 0; kk < kb; kk++) {
                            A_tile[i][kk] = m1[(i0 + i)*k + (k0 + kk)];
                        }
                    }
                    
                    for (int kk = 0; kk < kb; kk++) {
                        float* dest = B_tile[kk];
                        const float* src = &m2[(k0 + kk)*m + j0];
                        
                        std::memcpy(dest, src, nb * sizeof(float));
                    }
                    
                    for (int i = 0; i < mb; i += 4) {
                        const int ib = std::min(i + 4, mb);
                        if (ib - i == 4) {
                            
                            for (int j = 0; j < nb; j += 16) {
                                const int jb = std::min(j + 16, nb);
                                if (jb - j == 16) {
                                    
                                    micro_kernel_4x16(&A_tile[i][0], &B_tile[0][j], &C_tile[i][j], kb, KC, NC, NC);
                                } else {
                                    
                                    for (int ii = i; ii < ib; ii++) {
                                        for (int jj = j; jj < jb; jj++) {
                                            float sum = 0.0f;
                                            #pragma omp simd reduction(+:sum)
                                            for (int kk = 0; kk < kb; kk++) {
                                                sum += A_tile[ii][kk] * B_tile[kk][jj];
                                            }
                                            C_tile[ii][jj] += sum;
                                        }
                                    }
                                }
                            }
                        } else {
                            
                            for (int ii = i; ii < ib; ii++) {
                                for (int j = 0; j < nb; j++) {
                                    float sum = 0.0f;
                                    #pragma omp simd reduction(+:sum)
                                    for (int kk = 0; kk < kb; kk++) {
                                        sum += A_tile[ii][kk] * B_tile[kk][j];
                                    }
                                    C_tile[ii][j] += sum;
                                }
                            }
                        }
                    }
                }
                
                for (int i = 0; i < mb; i++) {
                    float* dest = &result[(i0 + i)*m + j0];
                    const float* src = &C_tile[i][0];
                    std::memcpy(dest, src, nb * sizeof(float));
                }
            }
        }
    }
    
    sol_fs.write(reinterpret_cast<const char*>(result.get()), sizeof(float) * n * m);
    sol_fs.close();
    return sol_path;
}

} 