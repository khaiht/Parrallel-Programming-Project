/**
 * ═══════════════════════════════════════════════════════════════════════════════
 *                    CUDA WATER SIMULATION (SPH - Smoothed Particle Hydrodynamics)
 * ═══════════════════════════════════════════════════════════════════════════════
 * 
 * Author: Water Simulation Project
 * Description: GPU-accelerated realistic water simulation using CUDA
 *              with SDL3 real-time visualization
 * 
 * Algorithm: SPH (Smoothed Particle Hydrodynamics) - Navier-Stokes based
 * 
 * Features:
 *   - Density and Pressure calculation using SPH kernels
 *   - Viscosity forces for smooth fluid motion
 *   - Surface tension for realistic water behavior
 *   - Gravity and boundary collision handling
 *   - Beautiful blue water visualization with foam/splash effects
 *   - Interactive controls (add water, reset, etc.)
 * 
 * Controls:
 *   ESC      - Exit
 *   SPACE    - Pause/Resume
 *   R        - Reset simulation
 *   Left Click - Add water particles
 *   +/-      - Increase/Decrease viscosity
 *   G        - Toggle gravity direction
 * 
 * ═══════════════════════════════════════════════════════════════════════════════
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <SDL3/SDL.h>

// ═══════════════════════════════════════════════════════════════════════════════
//                           SIMULATION PARAMETERS
// ═══════════════════════════════════════════════════════════════════════════════

// --- Number of particles ---
#define MAX_PARTICLES 50000
#define INITIAL_PARTICLES 40000

// --- SPH Physics Constants ---
#define REST_DENSITY    1000.0f     // Water rest density (kg/m³)
#define GAS_CONSTANT    2000.0f     // Stiffness constant for pressure
#define H               0.045f      // Smoothing radius (larger = smoother)
#define H2              (H * H)     // H squared
#define MASS            0.02f       // Particle mass
#define VISCOSITY       400.0f      // Dynamic viscosity coefficient (higher = smoother)
#define SURFACE_TENSION 0.0728f     // Surface tension coefficient
#define DT              0.0003f     // Time step (smaller = more stable)
#define GRAVITY         -9.81f      // Gravity acceleration

// --- Boundary Box ---
#define BOX_WIDTH       1.5f
#define BOX_HEIGHT      1.0f
#define BOX_DEPTH       0.3f
#define BOUNDARY_DAMPING 0.3f       // Velocity damping at boundaries

// --- CUDA Configuration ---
#define BLOCK_SIZE      256

// --- Visualization ---
#define WINDOW_WIDTH    1400
#define WINDOW_HEIGHT   900
#define RENDER_SCALE    600.0f

// --- SPH Kernel Constants (precomputed) ---
#define PI 3.14159265359f
#define POLY6_CONST     (315.0f / (64.0f * PI * powf(H, 9)))
#define SPIKY_CONST     (-45.0f / (PI * powf(H, 6)))
#define VISC_LAP_CONST  (45.0f / (PI * powf(H, 6)))

// --- Tile size for shared memory optimization ---
#define TILE_SIZE       128

// ═══════════════════════════════════════════════════════════════════════════════
//                         CONSTANT MEMORY (Fast read-only cache)
// ═══════════════════════════════════════════════════════════════════════════════
// Constant memory được cache và broadcast đến tất cả threads trong warp
// Tối ưu cho các giá trị được đọc bởi nhiều threads cùng lúc

__constant__ float d_H;              // Smoothing radius
__constant__ float d_H2;             // H squared
__constant__ float d_MASS;           // Particle mass
__constant__ float d_REST_DENSITY;   // Rest density
__constant__ float d_GAS_CONSTANT;   // Gas constant
__constant__ float d_VISCOSITY;      // Viscosity
__constant__ float d_GRAVITY;        // Gravity
__constant__ float d_POLY6;          // Poly6 kernel constant
__constant__ float d_SPIKY;          // Spiky kernel constant
__constant__ float d_VISC_LAP;       // Viscosity Laplacian constant
__constant__ float d_BOX_WIDTH;      // Box dimensions
__constant__ float d_BOX_HEIGHT;
__constant__ float d_BOX_DEPTH;
__constant__ float d_DAMPING;        // Boundary damping
__constant__ float d_DT;             // Time step

// ═══════════════════════════════════════════════════════════════════════════════
//                              DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

// Structure of Arrays (SoA) for GPU coalesced memory access
// Khi threads trong warp truy cập x[0], x[1], x[2]... -> coalesced access
// Nếu dùng AoS (Array of Structures) -> strided access -> chậm hơn nhiều
typedef struct {
    float* __restrict__ x;           // Position X (__restrict__ = no pointer aliasing)
    float* __restrict__ y;           // Position Y
    float* __restrict__ z;           // Position Z
    float* __restrict__ vx;          // Velocity X
    float* __restrict__ vy;          // Velocity Y
    float* __restrict__ vz;          // Velocity Z
    float* __restrict__ density;     // Computed density
    float* __restrict__ pressure;    // Computed pressure
    float* __restrict__ fx;          // Force X
    float* __restrict__ fy;          // Force Y
    float* __restrict__ fz;          // Force Z
} Particles;

// Camera for visualization
typedef struct {
    float offsetX;
    float offsetY;
    float scale;
} Camera;

// ═══════════════════════════════════════════════════════════════════════════════
//                              CUDA UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

#define CUDA_CHECK(call, msg) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s: %s\n", \
                __FILE__, __LINE__, msg, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ═══════════════════════════════════════════════════════════════════════════════
//                           MEMORY MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════════

void allocateParticlesHost(Particles* p, int n) {
    p->x        = (float*)malloc(n * sizeof(float));
    p->y        = (float*)malloc(n * sizeof(float));
    p->z        = (float*)malloc(n * sizeof(float));
    p->vx       = (float*)malloc(n * sizeof(float));
    p->vy       = (float*)malloc(n * sizeof(float));
    p->vz       = (float*)malloc(n * sizeof(float));
    p->density  = (float*)malloc(n * sizeof(float));
    p->pressure = (float*)malloc(n * sizeof(float));
    p->fx       = (float*)malloc(n * sizeof(float));
    p->fy       = (float*)malloc(n * sizeof(float));
    p->fz       = (float*)malloc(n * sizeof(float));
}

void freeParticlesHost(Particles* p) {
    free(p->x);  free(p->y);  free(p->z);
    free(p->vx); free(p->vy); free(p->vz);
    free(p->density); free(p->pressure);
    free(p->fx); free(p->fy); free(p->fz);
}

void allocateParticlesDevice(Particles* p, int n) {
    CUDA_CHECK(cudaMalloc(&p->x,        n * sizeof(float)), "malloc x");
    CUDA_CHECK(cudaMalloc(&p->y,        n * sizeof(float)), "malloc y");
    CUDA_CHECK(cudaMalloc(&p->z,        n * sizeof(float)), "malloc z");
    CUDA_CHECK(cudaMalloc(&p->vx,       n * sizeof(float)), "malloc vx");
    CUDA_CHECK(cudaMalloc(&p->vy,       n * sizeof(float)), "malloc vy");
    CUDA_CHECK(cudaMalloc(&p->vz,       n * sizeof(float)), "malloc vz");
    CUDA_CHECK(cudaMalloc(&p->density,  n * sizeof(float)), "malloc density");
    CUDA_CHECK(cudaMalloc(&p->pressure, n * sizeof(float)), "malloc pressure");
    CUDA_CHECK(cudaMalloc(&p->fx,       n * sizeof(float)), "malloc fx");
    CUDA_CHECK(cudaMalloc(&p->fy,       n * sizeof(float)), "malloc fy");
    CUDA_CHECK(cudaMalloc(&p->fz,       n * sizeof(float)), "malloc fz");
}

void freeParticlesDevice(Particles* p) {
    cudaFree(p->x);  cudaFree(p->y);  cudaFree(p->z);
    cudaFree(p->vx); cudaFree(p->vy); cudaFree(p->vz);
    cudaFree(p->density); cudaFree(p->pressure);
    cudaFree(p->fx); cudaFree(p->fy); cudaFree(p->fz);
}

void copyParticlesToDevice(Particles* dst, Particles* src, int n) {
    CUDA_CHECK(cudaMemcpy(dst->x,  src->x,  n * sizeof(float), cudaMemcpyHostToDevice), "H2D x");
    CUDA_CHECK(cudaMemcpy(dst->y,  src->y,  n * sizeof(float), cudaMemcpyHostToDevice), "H2D y");
    CUDA_CHECK(cudaMemcpy(dst->z,  src->z,  n * sizeof(float), cudaMemcpyHostToDevice), "H2D z");
    CUDA_CHECK(cudaMemcpy(dst->vx, src->vx, n * sizeof(float), cudaMemcpyHostToDevice), "H2D vx");
    CUDA_CHECK(cudaMemcpy(dst->vy, src->vy, n * sizeof(float), cudaMemcpyHostToDevice), "H2D vy");
    CUDA_CHECK(cudaMemcpy(dst->vz, src->vz, n * sizeof(float), cudaMemcpyHostToDevice), "H2D vz");
}

void copyParticlesToHost(Particles* dst, Particles* src, int n) {
    CUDA_CHECK(cudaMemcpy(dst->x,  src->x,  n * sizeof(float), cudaMemcpyDeviceToHost), "D2H x");
    CUDA_CHECK(cudaMemcpy(dst->y,  src->y,  n * sizeof(float), cudaMemcpyDeviceToHost), "D2H y");
    CUDA_CHECK(cudaMemcpy(dst->z,  src->z,  n * sizeof(float), cudaMemcpyDeviceToHost), "D2H z");
    CUDA_CHECK(cudaMemcpy(dst->vx, src->vx, n * sizeof(float), cudaMemcpyDeviceToHost), "D2H vx");
    CUDA_CHECK(cudaMemcpy(dst->vy, src->vy, n * sizeof(float), cudaMemcpyDeviceToHost), "D2H vy");
    CUDA_CHECK(cudaMemcpy(dst->vz, src->vz, n * sizeof(float), cudaMemcpyDeviceToHost), "D2H vz");
    CUDA_CHECK(cudaMemcpy(dst->density, src->density, n * sizeof(float), cudaMemcpyDeviceToHost), "D2H density");
}

// ═══════════════════════════════════════════════════════════════════════════════
//                         INITIAL CONDITIONS
// ═══════════════════════════════════════════════════════════════════════════════

void initWaterBlock(Particles* p, int* numParticles, int mode) {
    int count = 0;
    float spacing = H * 0.5f;  // Initial particle spacing
    
    if (mode == 1) {
        // Dam break scenario - water block on left side
        printf("  -> Initializing: DAM BREAK\n");
        for (float x = -BOX_WIDTH/2 + spacing; x < -BOX_WIDTH/4 && count < MAX_PARTICLES; x += spacing) {
            for (float y = -BOX_HEIGHT/2 + spacing; y < BOX_HEIGHT/2 - spacing && count < MAX_PARTICLES; y += spacing) {
                for (float z = -BOX_DEPTH/2 + spacing; z < BOX_DEPTH/2 - spacing && count < MAX_PARTICLES; z += spacing) {
                    p->x[count] = x + ((float)rand() / RAND_MAX - 0.5f) * spacing * 0.1f;
                    p->y[count] = y + ((float)rand() / RAND_MAX - 0.5f) * spacing * 0.1f;
                    p->z[count] = z + ((float)rand() / RAND_MAX - 0.5f) * spacing * 0.1f;
                    p->vx[count] = 0.0f;
                    p->vy[count] = 0.0f;
                    p->vz[count] = 0.0f;
                    count++;
                }
            }
        }
    } else if (mode == 2) {
        // Water drop into pool
        printf("  -> Initializing: WATER DROP INTO POOL\n");
        
        // Bottom pool
        for (float x = -BOX_WIDTH/2 + spacing; x < BOX_WIDTH/2 - spacing && count < MAX_PARTICLES * 0.6f; x += spacing) {
            for (float y = -BOX_HEIGHT/2 + spacing; y < -BOX_HEIGHT/4 && count < MAX_PARTICLES * 0.6f; y += spacing) {
                for (float z = -BOX_DEPTH/2 + spacing; z < BOX_DEPTH/2 - spacing && count < MAX_PARTICLES * 0.6f; z += spacing) {
                    p->x[count] = x;
                    p->y[count] = y;
                    p->z[count] = z;
                    p->vx[count] = 0.0f;
                    p->vy[count] = 0.0f;
                    p->vz[count] = 0.0f;
                    count++;
                }
            }
        }
        
        // Dropping sphere
        float dropCenterX = 0.0f;
        float dropCenterY = BOX_HEIGHT/3;
        float dropRadius = 0.15f;
        
        for (float x = dropCenterX - dropRadius; x < dropCenterX + dropRadius && count < MAX_PARTICLES; x += spacing) {
            for (float y = dropCenterY - dropRadius; y < dropCenterY + dropRadius && count < MAX_PARTICLES; y += spacing) {
                for (float z = -dropRadius; z < dropRadius && count < MAX_PARTICLES; z += spacing) {
                    float dx = x - dropCenterX;
                    float dy = y - dropCenterY;
                    float dz = z;
                    if (dx*dx + dy*dy + dz*dz < dropRadius*dropRadius) {
                        p->x[count] = x;
                        p->y[count] = y;
                        p->z[count] = z;
                        p->vx[count] = 0.0f;
                        p->vy[count] = -2.0f;  // Initial downward velocity
                        p->vz[count] = 0.0f;
                        count++;
                    }
                }
            }
        }
    } else {
        // Double dam break (from both sides)
        printf("  -> Initializing: DOUBLE DAM BREAK\n");
        
        // Left block
        for (float x = -BOX_WIDTH/2 + spacing; x < -BOX_WIDTH/3 && count < MAX_PARTICLES/2; x += spacing) {
            for (float y = -BOX_HEIGHT/2 + spacing; y < BOX_HEIGHT/3 && count < MAX_PARTICLES/2; y += spacing) {
                for (float z = -BOX_DEPTH/2 + spacing; z < BOX_DEPTH/2 - spacing && count < MAX_PARTICLES/2; z += spacing) {
                    p->x[count] = x;
                    p->y[count] = y;
                    p->z[count] = z;
                    p->vx[count] = 0.0f;
                    p->vy[count] = 0.0f;
                    p->vz[count] = 0.0f;
                    count++;
                }
            }
        }
        
        // Right block
        int halfCount = count;
        for (float x = BOX_WIDTH/3; x < BOX_WIDTH/2 - spacing && count < MAX_PARTICLES; x += spacing) {
            for (float y = -BOX_HEIGHT/2 + spacing; y < BOX_HEIGHT/3 && count < MAX_PARTICLES; y += spacing) {
                for (float z = -BOX_DEPTH/2 + spacing; z < BOX_DEPTH/2 - spacing && count < MAX_PARTICLES; z += spacing) {
                    p->x[count] = x;
                    p->y[count] = y;
                    p->z[count] = z;
                    p->vx[count] = 0.0f;
                    p->vy[count] = 0.0f;
                    p->vz[count] = 0.0f;
                    count++;
                }
            }
        }
    }
    
    *numParticles = count;
    printf("  -> Created %d water particles\n", count);
}

// ═══════════════════════════════════════════════════════════════════════════════
//                         SPH CUDA KERNELS (OPTIMIZED)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Poly6 Kernel - Used for density calculation
 * Sử dụng __forceinline__ để compiler inline hàm này
 * Giảm overhead gọi hàm
 */
__device__ __forceinline__ float poly6Kernel(float r2) {
    if (r2 >= d_H2) return 0.0f;
    float diff = d_H2 - r2;
    return d_POLY6 * diff * diff * diff;
}

/**
 * Spiky Kernel Gradient - Used for pressure forces
 * Sử dụng rsqrtf() (reciprocal sqrt) - nhanh hơn 1/sqrt()
 */
__device__ __forceinline__ float spikyKernelGrad(float r) {
    if (r >= d_H || r < 1e-6f) return 0.0f;
    float diff = d_H - r;
    return d_SPIKY * diff * diff;
}

/**
 * Viscosity Kernel Laplacian - Used for viscosity forces
 */
__device__ __forceinline__ float viscosityKernelLap(float r) {
    if (r >= d_H) return 0.0f;
    return d_VISC_LAP * (d_H - r);
}

/**
 * KERNEL 1: Compute Density and Pressure (OPTIMIZED with Shared Memory Tiling)
 * 
 * Tối ưu hóa:
 * 1. Shared Memory Tiling - Load tile của particles vào shared memory
 * 2. Giảm global memory access từ O(N²) xuống O(N²/TILE_SIZE) cho mỗi thread
 * 3. __syncthreads() đảm bảo tất cả threads load xong trước khi tính toán
 * 4. Loop unrolling với #pragma unroll
 */
__global__ void computeDensityPressureKernel(
    const float* __restrict__ x, 
    const float* __restrict__ y, 
    const float* __restrict__ z,
    float* __restrict__ density, 
    float* __restrict__ pressure,
    int n
) {
    // Shared memory cho tiling - cache vị trí của các particles trong tile
    __shared__ float tile_x[TILE_SIZE];
    __shared__ float tile_y[TILE_SIZE];
    __shared__ float tile_z[TILE_SIZE];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load vị trí particle i vào registers (nhanh nhất)
    float xi, yi, zi;
    if (i < n) {
        xi = x[i];
        yi = y[i];
        zi = z[i];
    }
    
    float rho = 0.0f;
    
    // Tính số tiles cần xử lý
    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    
    // Duyệt qua từng tile
    for (int tile = 0; tile < numTiles; tile++) {
        // Index của particle trong tile hiện tại mà thread này sẽ load
        int loadIdx = tile * TILE_SIZE + threadIdx.x;
        
        // Collaborative loading: mỗi thread load 1 particle vào shared memory
        // Đây là coalesced access vì các threads liên tiếp load các vị trí liên tiếp
        if (loadIdx < n && threadIdx.x < TILE_SIZE) {
            tile_x[threadIdx.x] = x[loadIdx];
            tile_y[threadIdx.x] = y[loadIdx];
            tile_z[threadIdx.x] = z[loadIdx];
        }
        
        // BARRIER: Đợi tất cả threads load xong
        // Quan trọng: Không có barrier này sẽ có race condition!
        __syncthreads();
        
        // Tính density contribution từ particles trong tile này
        if (i < n) {
            // Số particles thực sự trong tile này
            int tileEnd = min(TILE_SIZE, n - tile * TILE_SIZE);
            
            // Loop unrolling: Compiler sẽ unroll vòng lặp này
            // Giảm branch overhead và tăng ILP (Instruction Level Parallelism)
            #pragma unroll 8
            for (int j = 0; j < tileEnd; j++) {
                float dx = xi - tile_x[j];
                float dy = yi - tile_y[j];
                float dz = zi - tile_z[j];
                
                // fmaf(a,b,c) = a*b + c, thực hiện trong 1 instruction
                // Nhanh hơn và chính xác hơn a*b + c
                float r2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
                
                rho += d_MASS * poly6Kernel(r2);
            }
        }
        
        // BARRIER: Đợi tất cả threads tính xong trước khi load tile mới
        __syncthreads();
    }
    
    if (i < n) {
        density[i] = rho;
        
        // Tait equation of state
        float p = d_GAS_CONSTANT * (rho - d_REST_DENSITY);
        
        // Branchless max: tránh branch divergence trong warp
        pressure[i] = fmaxf(p, 0.0f);
    }
}

/**
 * KERNEL 2: Compute Forces (OPTIMIZED with Shared Memory Tiling)
 * 
 * Tối ưu hóa bổ sung:
 * 1. rsqrtf() thay vì 1/sqrt() - hardware accelerated
 * 2. Prefetch velocity vào shared memory
 * 3. Register reuse để giảm memory traffic
 */
__global__ void computeForcesKernel(
    const float* __restrict__ x, 
    const float* __restrict__ y, 
    const float* __restrict__ z,
    const float* __restrict__ vx, 
    const float* __restrict__ vy, 
    const float* __restrict__ vz,
    const float* __restrict__ density, 
    const float* __restrict__ pressure,
    float* __restrict__ fx, 
    float* __restrict__ fy, 
    float* __restrict__ fz,
    int n
) {
    // Shared memory tiles - bao gồm cả velocity để giảm global memory access
    __shared__ float tile_x[TILE_SIZE];
    __shared__ float tile_y[TILE_SIZE];
    __shared__ float tile_z[TILE_SIZE];
    __shared__ float tile_vx[TILE_SIZE];
    __shared__ float tile_vy[TILE_SIZE];
    __shared__ float tile_vz[TILE_SIZE];
    __shared__ float tile_density[TILE_SIZE];
    __shared__ float tile_pressure[TILE_SIZE];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load particle i data vào registers
    float xi, yi, zi, vxi, vyi, vzi, rhoi, pi;
    if (i < n) {
        xi = x[i];
        yi = y[i];
        zi = z[i];
        vxi = vx[i];
        vyi = vy[i];
        vzi = vz[i];
        rhoi = density[i];
        pi = pressure[i];
    }
    
    // Accumulate forces trong registers
    float forceX = 0.0f;
    float forceY = 0.0f;
    float forceZ = 0.0f;
    
    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < numTiles; tile++) {
        int loadIdx = tile * TILE_SIZE + threadIdx.x;
        
        // Collaborative loading với coalesced access
        if (loadIdx < n && threadIdx.x < TILE_SIZE) {
            tile_x[threadIdx.x] = x[loadIdx];
            tile_y[threadIdx.x] = y[loadIdx];
            tile_z[threadIdx.x] = z[loadIdx];
            tile_vx[threadIdx.x] = vx[loadIdx];
            tile_vy[threadIdx.x] = vy[loadIdx];
            tile_vz[threadIdx.x] = vz[loadIdx];
            tile_density[threadIdx.x] = density[loadIdx];
            tile_pressure[threadIdx.x] = pressure[loadIdx];
        }
        
        __syncthreads();
        
        if (i < n) {
            int tileEnd = min(TILE_SIZE, n - tile * TILE_SIZE);
            int globalJ = tile * TILE_SIZE;
            
            #pragma unroll 4
            for (int j = 0; j < tileEnd; j++) {
                // Skip self-interaction
                if (globalJ + j == i) continue;
                
                float dx = xi - tile_x[j];
                float dy = yi - tile_y[j];
                float dz = zi - tile_z[j];
                float r2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
                
                // Early exit nếu quá xa - giảm computation
                if (r2 >= d_H2 || r2 < 1e-12f) continue;
                
                // rsqrtf = 1/sqrt, hardware accelerated trên GPU
                // Nhanh hơn nhiều so với sqrt rồi chia
                float r = rsqrtf(r2);
                r = 1.0f / r;  // Convert back to r (rsqrtf returns 1/sqrt)
                
                float rhoj = tile_density[j];
                float pj = tile_pressure[j];
                
                // Tính reciprocal một lần, dùng nhiều lần
                float inv_rhoj = 1.0f / rhoj;
                float inv_r = 1.0f / r;
                
                // Pressure force (symmetric formulation)
                float pressureForce = -d_MASS * (pi + pj) * 0.5f * inv_rhoj * spikyKernelGrad(r);
                
                // Dùng fmaf cho multiply-add operations
                forceX = fmaf(pressureForce * inv_r, dx, forceX);
                forceY = fmaf(pressureForce * inv_r, dy, forceY);
                forceZ = fmaf(pressureForce * inv_r, dz, forceZ);
                
                // Viscosity force
                float viscForce = d_VISCOSITY * d_MASS * inv_rhoj * viscosityKernelLap(r);
                forceX = fmaf(viscForce, tile_vx[j] - vxi, forceX);
                forceY = fmaf(viscForce, tile_vy[j] - vyi, forceY);
                forceZ = fmaf(viscForce, tile_vz[j] - vzi, forceZ);
            }
        }
        
        __syncthreads();
    }
    
    if (i < n) {
        // Gravity force
        forceY = fmaf(rhoi, d_GRAVITY, forceY);
        
        fx[i] = forceX;
        fy[i] = forceY;
        fz[i] = forceZ;
    }
}

/**
 * KERNEL 3: Integrate (OPTIMIZED)
 * 
 * Tối ưu:
 * 1. Branchless boundary handling với fminf/fmaxf
 * 2. Register-based computation
 * 3. Single write to global memory per array
 */
__global__ void integrateKernel(
    float* __restrict__ x, 
    float* __restrict__ y, 
    float* __restrict__ z,
    float* __restrict__ vx, 
    float* __restrict__ vy, 
    float* __restrict__ vz,
    const float* __restrict__ fx, 
    const float* __restrict__ fy, 
    const float* __restrict__ fz,
    const float* __restrict__ density,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Load vào registers
    float rhoi = density[i];
    rhoi = fmaxf(rhoi, 1e-6f);  // Branchless safety check
    
    float inv_rho = 1.0f / rhoi;  // Tính 1 lần, dùng 3 lần
    
    // Acceleration = Force / Density
    float ax = fx[i] * inv_rho;
    float ay = fy[i] * inv_rho;
    float az = fz[i] * inv_rho;
    
    // Clamp acceleration - branchless với fminf/fmaxf
    float accMag2 = fmaf(ax, ax, fmaf(ay, ay, az * az));
    const float maxAcc = 500.0f;
    const float maxAcc2 = maxAcc * maxAcc;
    
    if (accMag2 > maxAcc2) {
        float scale = maxAcc * rsqrtf(accMag2);  // rsqrtf nhanh hơn
        ax *= scale;
        ay *= scale;
        az *= scale;
    }
    
    // Load current velocity
    float vxi = vx[i];
    float vyi = vy[i];
    float vzi = vz[i];
    
    // Update velocity: v += a * dt
    vxi = fmaf(ax, d_DT, vxi);
    vyi = fmaf(ay, d_DT, vyi);
    vzi = fmaf(az, d_DT, vzi);
    
    // Load current position
    float xi = x[i];
    float yi = y[i];
    float zi = z[i];
    
    // Update position: x += v * dt
    xi = fmaf(vxi, d_DT, xi);
    yi = fmaf(vyi, d_DT, yi);
    zi = fmaf(vzi, d_DT, zi);
    
    // Boundary conditions - OPTIMIZED với branchless operations
    float halfWidth = d_BOX_WIDTH * 0.5f;
    float halfHeight = d_BOX_HEIGHT * 0.5f;
    float halfDepth = d_BOX_DEPTH * 0.5f;
    
    // X boundaries
    if (xi < -halfWidth) {
        xi = -halfWidth;
        vxi *= -d_DAMPING;
    } else if (xi > halfWidth) {
        xi = halfWidth;
        vxi *= -d_DAMPING;
    }
    
    // Y boundaries
    if (yi < -halfHeight) {
        yi = -halfHeight;
        vyi *= -d_DAMPING;
    } else if (yi > halfHeight) {
        yi = halfHeight;
        vyi *= -d_DAMPING;
    }
    
    // Z boundaries
    if (zi < -halfDepth) {
        zi = -halfDepth;
        vzi *= -d_DAMPING;
    } else if (zi > halfDepth) {
        zi = halfDepth;
        vzi *= -d_DAMPING;
    }
    
    // Write back - coalesced writes
    x[i] = xi;
    y[i] = yi;
    z[i] = zi;
    vx[i] = vxi;
    vy[i] = vyi;
    vz[i] = vzi;
}

// ═══════════════════════════════════════════════════════════════════════════════
//                              RENDERING
// ═══════════════════════════════════════════════════════════════════════════════

void renderWater(SDL_Renderer* renderer, Particles* p, int n, Camera* cam) {
    // Draw boundary box
    int boxLeft   = (int)((-BOX_WIDTH/2) * cam->scale + WINDOW_WIDTH/2 + cam->offsetX);
    int boxRight  = (int)((BOX_WIDTH/2) * cam->scale + WINDOW_WIDTH/2 + cam->offsetX);
    int boxTop    = (int)((-BOX_HEIGHT/2) * cam->scale + WINDOW_HEIGHT/2 + cam->offsetY);
    int boxBottom = (int)((BOX_HEIGHT/2) * cam->scale + WINDOW_HEIGHT/2 + cam->offsetY);
    
    // Flip Y for screen coordinates
    int temp = boxTop;
    boxTop = WINDOW_HEIGHT - boxBottom;
    boxBottom = WINDOW_HEIGHT - temp;
    
    SDL_SetRenderDrawColor(renderer, 100, 100, 100, 255);
    SDL_RenderLine(renderer, (float)boxLeft, (float)boxTop, (float)boxRight, (float)boxTop);
    SDL_RenderLine(renderer, (float)boxRight, (float)boxTop, (float)boxRight, (float)boxBottom);
    SDL_RenderLine(renderer, (float)boxRight, (float)boxBottom, (float)boxLeft, (float)boxBottom);
    SDL_RenderLine(renderer, (float)boxLeft, (float)boxBottom, (float)boxLeft, (float)boxTop);
    
    // Find velocity range for color mapping
    float maxVel = 0.1f;  // Minimum to avoid division by zero
    for (int i = 0; i < n; i++) {
        float vel = sqrtf(p->vx[i]*p->vx[i] + p->vy[i]*p->vy[i] + p->vz[i]*p->vz[i]);
        if (vel > maxVel) maxVel = vel;
    }
    
    // Render particles
    for (int i = 0; i < n; i++) {
        int sx = (int)(p->x[i] * cam->scale + WINDOW_WIDTH/2 + cam->offsetX);
        int sy = WINDOW_HEIGHT - (int)(p->y[i] * cam->scale + WINDOW_HEIGHT/2 + cam->offsetY);
        
        // Skip off-screen particles
        if (sx < 0 || sx >= WINDOW_WIDTH || sy < 0 || sy >= WINDOW_HEIGHT) continue;
        
        // Color based on velocity (blue -> cyan -> white for splashing)
        float vel = sqrtf(p->vx[i]*p->vx[i] + p->vy[i]*p->vy[i] + p->vz[i]*p->vz[i]);
        float velNorm = vel / maxVel;
        
        // Density-based alpha (denser = more opaque)
        float densityNorm = p->density[i] / (REST_DENSITY * 1.5f);
        if (densityNorm > 1.0f) densityNorm = 1.0f;
        
        // Blue water gradient
        int r, g, b;
        if (velNorm < 0.3f) {
            // Calm water - deep blue
            r = 20 + (int)(30 * densityNorm);
            g = 80 + (int)(50 * densityNorm);
            b = 180 + (int)(50 * densityNorm);
        } else if (velNorm < 0.6f) {
            // Moving water - lighter blue
            float t = (velNorm - 0.3f) / 0.3f;
            r = 50 + (int)(100 * t);
            g = 130 + (int)(70 * t);
            b = 200 + (int)(30 * t);
        } else {
            // Fast moving / splash - cyan to white
            float t = (velNorm - 0.6f) / 0.4f;
            r = 150 + (int)(105 * t);
            g = 200 + (int)(55 * t);
            b = 230 + (int)(25 * t);
        }
        
        SDL_SetRenderDrawColor(renderer, r, g, b, 255);
        
        // Draw particle as small filled circle (2-3 pixels radius for realistic look)
        int radius = 2;
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                if (dx*dx + dy*dy <= radius*radius) {
                    SDL_RenderPoint(renderer, (float)(sx + dx), (float)(sy + dy));
                }
            }
        }
    }
}

void renderUI(SDL_Renderer* renderer, int numParticles, float simTime, float fps, bool paused, int mode) {
    // Render simple UI overlay info using colored bars
    // Top bar showing simulation state
    
    if (paused) {
        // Red bar for paused
        SDL_SetRenderDrawColor(renderer, 255, 100, 100, 200);
        SDL_FRect pauseBar = {10, 10, 100, 10};
        SDL_RenderFillRect(renderer, &pauseBar);
    } else {
        // Green bar for running
        SDL_SetRenderDrawColor(renderer, 100, 255, 100, 200);
        SDL_FRect runBar = {10, 10, 100, 10};
        SDL_RenderFillRect(renderer, &runBar);
    }
    
    // FPS indicator bar (blue)
    float fpsNorm = fps / 60.0f;
    if (fpsNorm > 1.0f) fpsNorm = 1.0f;
    SDL_SetRenderDrawColor(renderer, 100, 100, 255, 200);
    SDL_FRect fpsBar = {10, 25, fpsNorm * 100, 10};
    SDL_RenderFillRect(renderer, &fpsBar);
}

// ═══════════════════════════════════════════════════════════════════════════════
//                              MAIN FUNCTION
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Khởi tạo Constant Memory
 * Copy các hằng số từ host sang constant memory trên GPU
 * Constant memory được cache và broadcast hiệu quả đến tất cả threads
 */
void initConstantMemory() {
    float h = H;
    float h2 = H2;
    float mass = MASS;
    float restDensity = REST_DENSITY;
    float gasConstant = GAS_CONSTANT;
    float viscosity = VISCOSITY;
    float gravity = GRAVITY;
    float poly6 = POLY6_CONST;
    float spiky = SPIKY_CONST;
    float viscLap = VISC_LAP_CONST;
    float boxWidth = BOX_WIDTH;
    float boxHeight = BOX_HEIGHT;
    float boxDepth = BOX_DEPTH;
    float damping = BOUNDARY_DAMPING;
    float dt = DT;
    
    CUDA_CHECK(cudaMemcpyToSymbol(d_H, &h, sizeof(float)), "const H");
    CUDA_CHECK(cudaMemcpyToSymbol(d_H2, &h2, sizeof(float)), "const H2");
    CUDA_CHECK(cudaMemcpyToSymbol(d_MASS, &mass, sizeof(float)), "const MASS");
    CUDA_CHECK(cudaMemcpyToSymbol(d_REST_DENSITY, &restDensity, sizeof(float)), "const REST_DENSITY");
    CUDA_CHECK(cudaMemcpyToSymbol(d_GAS_CONSTANT, &gasConstant, sizeof(float)), "const GAS_CONSTANT");
    CUDA_CHECK(cudaMemcpyToSymbol(d_VISCOSITY, &viscosity, sizeof(float)), "const VISCOSITY");
    CUDA_CHECK(cudaMemcpyToSymbol(d_GRAVITY, &gravity, sizeof(float)), "const GRAVITY");
    CUDA_CHECK(cudaMemcpyToSymbol(d_POLY6, &poly6, sizeof(float)), "const POLY6");
    CUDA_CHECK(cudaMemcpyToSymbol(d_SPIKY, &spiky, sizeof(float)), "const SPIKY");
    CUDA_CHECK(cudaMemcpyToSymbol(d_VISC_LAP, &viscLap, sizeof(float)), "const VISC_LAP");
    CUDA_CHECK(cudaMemcpyToSymbol(d_BOX_WIDTH, &boxWidth, sizeof(float)), "const BOX_WIDTH");
    CUDA_CHECK(cudaMemcpyToSymbol(d_BOX_HEIGHT, &boxHeight, sizeof(float)), "const BOX_HEIGHT");
    CUDA_CHECK(cudaMemcpyToSymbol(d_BOX_DEPTH, &boxDepth, sizeof(float)), "const BOX_DEPTH");
    CUDA_CHECK(cudaMemcpyToSymbol(d_DAMPING, &damping, sizeof(float)), "const DAMPING");
    CUDA_CHECK(cudaMemcpyToSymbol(d_DT, &dt, sizeof(float)), "const DT");
    
    printf("  ✓ Constant memory initialized\n");
}

int main(int argc, char* argv[]) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║        CUDA WATER SIMULATION - SPH (OPTIMIZED for Parallel Programming)      ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Optimizations Applied:                                                       ║\n");
    printf("║    • Constant Memory - Fast broadcast of SPH constants                        ║\n");
    printf("║    • Shared Memory Tiling - Reduced global memory access                      ║\n");
    printf("║    • Loop Unrolling - Reduced branch overhead                                 ║\n");
    printf("║    • Intrinsic Functions - rsqrtf, fmaf for speed                             ║\n");
    printf("║    • Memory Coalescing - Structure of Arrays (SoA)                            ║\n");
    printf("║    • __restrict__ pointers - No aliasing optimization                         ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Controls:                                                                    ║\n");
    printf("║    ESC       - Exit                                                           ║\n");
    printf("║    SPACE     - Pause/Resume                                                   ║\n");
    printf("║    R         - Reset simulation                                               ║\n");
    printf("║    1/2/3     - Switch scenario (Dam break / Drop / Double dam)                ║\n");
    printf("║    +/-       - Zoom in/out                                                    ║\n");
    printf("║    Arrows    - Pan camera                                                     ║\n");
    printf("║    Left Click- Add water particles                                            ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════════════╝\n\n");
    
    // --- Print CUDA Device Info ---
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("ERROR: No CUDA-capable GPU found!\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("CUDA Cores: %d\n", prop.multiProcessorCount * 128);
    printf("Global Memory: %.0f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("\n");
    
    // --- Initialize SDL ---
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL Error: %s\n", SDL_GetError());
        return 1;
    }
    
    SDL_Window* window = SDL_CreateWindow(
        "CUDA Water Simulation - SPH",
        WINDOW_WIDTH, WINDOW_HEIGHT,
        0
    );
    
    if (!window) {
        printf("Window Error: %s\n", SDL_GetError());
        return 1;
    }
    
    SDL_Renderer* renderer = SDL_CreateRenderer(window, NULL);
    if (!renderer) {
        printf("Renderer Error: %s\n", SDL_GetError());
        return 1;
    }
    
    // Enable alpha blending
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    
    // --- Initialize Constant Memory (GPU optimization) ---
    printf("Initializing GPU optimizations:\n");
    initConstantMemory();
    
    // --- Allocate Memory ---
    Particles hostParticles, deviceParticles;
    allocateParticlesHost(&hostParticles, MAX_PARTICLES);
    allocateParticlesDevice(&deviceParticles, MAX_PARTICLES);
    printf("  ✓ Memory allocated (Host + Device)\n");
    
    // --- Initialize Particles ---
    srand((unsigned int)time(NULL));
    int numParticles = 0;
    int currentMode = 1;
    initWaterBlock(&hostParticles, &numParticles, currentMode);
    copyParticlesToDevice(&deviceParticles, &hostParticles, numParticles);
    printf("  ✓ Particles initialized and copied to GPU\n\n");
    
    printf("Simulation Parameters:\n");
    printf("  Particles: %d\n", numParticles);
    printf("  Rest Density: %.1f kg/m³\n", REST_DENSITY);
    printf("  Smoothing Radius: %.3f m\n", H);
    printf("  Time Step: %.5f s\n", DT);
    printf("  Viscosity: %.1f\n", VISCOSITY);
    printf("  Tile Size: %d (Shared Memory)\n", TILE_SIZE);
    printf("  Block Size: %d threads\n", BLOCK_SIZE);
    printf("\nStarting OPTIMIZED simulation...\n\n");
    
    // --- Camera ---
    Camera camera = {0.0f, -100.0f, RENDER_SCALE};
    
    // --- CUDA Grid Configuration ---
    int numBlocks = (numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // --- Timing ---
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    
    // --- Main Loop ---
    bool running = true;
    bool paused = false;
    SDL_Event event;
    
    Uint64 frameStart = SDL_GetPerformanceCounter();
    Uint64 freq = SDL_GetPerformanceFrequency();
    int frameCount = 0;
    float fps = 60.0f;
    float totalSimTime = 0.0f;
    
    int subSteps = 6;  // Multiple sub-steps per frame for stability (higher = smoother)
    
    while (running) {
        // --- Handle Events ---
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) {
                running = false;
            } else if (event.type == SDL_EVENT_KEY_DOWN) {
                switch (event.key.key) {
                    case SDLK_ESCAPE:
                        running = false;
                        break;
                    case SDLK_SPACE:
                        paused = !paused;
                        printf("%s\n", paused ? "PAUSED" : "RESUMED");
                        break;
                    case SDLK_R:
                        printf("Resetting simulation...\n");
                        initWaterBlock(&hostParticles, &numParticles, currentMode);
                        copyParticlesToDevice(&deviceParticles, &hostParticles, numParticles);
                        numBlocks = (numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE;
                        totalSimTime = 0.0f;
                        break;
                    case SDLK_1:
                        currentMode = 1;
                        printf("Mode: Dam Break\n");
                        initWaterBlock(&hostParticles, &numParticles, currentMode);
                        copyParticlesToDevice(&deviceParticles, &hostParticles, numParticles);
                        numBlocks = (numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE;
                        totalSimTime = 0.0f;
                        break;
                    case SDLK_2:
                        currentMode = 2;
                        printf("Mode: Water Drop\n");
                        initWaterBlock(&hostParticles, &numParticles, currentMode);
                        copyParticlesToDevice(&deviceParticles, &hostParticles, numParticles);
                        numBlocks = (numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE;
                        totalSimTime = 0.0f;
                        break;
                    case SDLK_3:
                        currentMode = 3;
                        printf("Mode: Double Dam Break\n");
                        initWaterBlock(&hostParticles, &numParticles, currentMode);
                        copyParticlesToDevice(&deviceParticles, &hostParticles, numParticles);
                        numBlocks = (numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE;
                        totalSimTime = 0.0f;
                        break;
                    case SDLK_EQUALS:  // +
                        camera.scale *= 1.1f;
                        break;
                    case SDLK_MINUS:   // -
                        camera.scale *= 0.9f;
                        break;
                    case SDLK_UP:
                        camera.offsetY -= 20;
                        break;
                    case SDLK_DOWN:
                        camera.offsetY += 20;
                        break;
                    case SDLK_LEFT:
                        camera.offsetX += 20;
                        break;
                    case SDLK_RIGHT:
                        camera.offsetX -= 20;
                        break;
                }
            } else if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
                if (event.button.button == SDL_BUTTON_LEFT && numParticles < MAX_PARTICLES - 100) {
                    // Add water particles at click position
                    float clickX = (event.button.x - WINDOW_WIDTH/2 - camera.offsetX) / camera.scale;
                    float clickY = -(event.button.y - WINDOW_HEIGHT/2 - camera.offsetY) / camera.scale;
                    
                    // Add small cluster of particles
                    int addCount = 0;
                    float spacing = H * 0.5f;
                    for (float dx = -0.05f; dx <= 0.05f && numParticles + addCount < MAX_PARTICLES; dx += spacing) {
                        for (float dy = -0.05f; dy <= 0.05f && numParticles + addCount < MAX_PARTICLES; dy += spacing) {
                            int idx = numParticles + addCount;
                            hostParticles.x[idx] = clickX + dx;
                            hostParticles.y[idx] = clickY + dy;
                            hostParticles.z[idx] = 0.0f;
                            hostParticles.vx[idx] = 0.0f;
                            hostParticles.vy[idx] = 0.0f;
                            hostParticles.vz[idx] = 0.0f;
                            addCount++;
                        }
                    }
                    numParticles += addCount;
                    copyParticlesToDevice(&deviceParticles, &hostParticles, numParticles);
                    numBlocks = (numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE;
                    printf("Added %d particles (Total: %d)\n", addCount, numParticles);
                }
            }
        }
        
        // --- Physics Update (OPTIMIZED KERNELS) ---
        if (!paused) {
            cudaEventRecord(startEvent);
            
            for (int step = 0; step < subSteps; step++) {
                // Step 1: Compute density and pressure (Shared Memory Tiling)
                // Sử dụng constant memory cho các hằng số SPH
                computeDensityPressureKernel<<<numBlocks, BLOCK_SIZE>>>(
                    deviceParticles.x, deviceParticles.y, deviceParticles.z,
                    deviceParticles.density, deviceParticles.pressure,
                    numParticles
                );
                
                // Step 2: Compute forces (Shared Memory Tiling + Intrinsics)
                computeForcesKernel<<<numBlocks, BLOCK_SIZE>>>(
                    deviceParticles.x, deviceParticles.y, deviceParticles.z,
                    deviceParticles.vx, deviceParticles.vy, deviceParticles.vz,
                    deviceParticles.density, deviceParticles.pressure,
                    deviceParticles.fx, deviceParticles.fy, deviceParticles.fz,
                    numParticles
                );
                
                // Step 3: Integrate (Optimized with fmaf, rsqrtf)
                integrateKernel<<<numBlocks, BLOCK_SIZE>>>(
                    deviceParticles.x, deviceParticles.y, deviceParticles.z,
                    deviceParticles.vx, deviceParticles.vy, deviceParticles.vz,
                    deviceParticles.fx, deviceParticles.fy, deviceParticles.fz,
                    deviceParticles.density,
                    numParticles
                );
                
                totalSimTime += DT;
            }
            
            cudaEventRecord(stopEvent);
            cudaEventSynchronize(stopEvent);
        }
        
        // Copy back for rendering
        copyParticlesToHost(&hostParticles, &deviceParticles, numParticles);
        
        // --- Render ---
        SDL_SetRenderDrawColor(renderer, 10, 15, 30, 255);  // Dark blue background
        SDL_RenderClear(renderer);
        
        renderWater(renderer, &hostParticles, numParticles, &camera);
        renderUI(renderer, numParticles, totalSimTime, fps, paused, currentMode);
        
        SDL_RenderPresent(renderer);
        
        // --- FPS Calculation ---
        frameCount++;
        Uint64 now = SDL_GetPerformanceCounter();
        float elapsed = (float)(now - frameStart) / freq;
        if (elapsed >= 1.0f) {
            fps = frameCount / elapsed;
            frameCount = 0;
            frameStart = now;
            
            // Update window title
            char title[256];
            sprintf(title, "CUDA Water Simulation | Particles: %d | FPS: %.1f | Time: %.2fs | Mode: %d", 
                    numParticles, fps, totalSimTime, currentMode);
            SDL_SetWindowTitle(window, title);
        }
    }
    
    // --- Cleanup ---
    printf("\n\n=== SIMULATION ENDED ===\n");
    printf("Total simulated time: %.2f seconds\n", totalSimTime);
    printf("Particles: %d\n", numParticles);
    
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    
    freeParticlesHost(&hostParticles);
    freeParticlesDevice(&deviceParticles);
    
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    
    return 0;
}
