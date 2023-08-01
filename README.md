# GPU-COMPUTING

The project consists of developing the PCA (Principal Component Analysis) technique using CUDA runtime API in C. 
The purpose of the project is to show the speedup and the correctness obtained with parallel solution compared to the sequential solution.<br><br>
For more information, read the report.

## Setup
To run the project in Google Colab, clone the repository:
```
!git clone https://github.com/MattiaFerraretto/GPU-COMPUTING
```

To compile and run the main, copy and paste into a colab cell:
```
!nvcc -arch=sm_75 GPU-COMPUTING/main.cu GPU-COMPUTING/cudalib/cudalinalg.cu GPU-COMPUTING/clib/ndarray.cpp GPU-COMPUTING/clib/linalg.cpp -o pca && ./pca
```

To compile and run unit tests, copy and paste into a colab cell:
```
!nvcc -arch=sm_75 GPU-COMPUTING/test.cu GPU-COMPUTING/cudalib/cudalinalg.cu GPU-COMPUTING/clib/ndarray.cpp GPU-COMPUTING/clib/linalg.cpp -o test && ./test
```

To show kernel profiling results, copy and paste into a colab cell:
```
!nvcc -arch=sm_75 GPU-COMPUTING/profiling.cu GPU-COMPUTING/cudalib/cudalinalg.cu GPU-COMPUTING/clib/ndarray.cpp GPU-COMPUTING/clib/linalg.cpp -o profiling
!ncu --metrics gpu__time_active.avg,smsp__sass_average_branch_targets_threads_uniform.pct,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio ./profiling
```
