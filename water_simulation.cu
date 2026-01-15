/*
Phím điều khiển:
ESC              : Thoát
Click chuột trái : Thêm nước
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <SDL3/SDL.h>

//===========================================================================
// THAM SỐ CẤU HÌNH MÔ PHỎNG
//===========================================================================

// Số lượng hạt nước
#define MAX_PARTICLES 50000         // Số hạt tối đa
#define INITIAL_PARTICLES 18000     // Số hạt ban đầu

// Các hằng số vật lý cho SPH
#define REST_DENSITY    1000.0f     // Mật độ nước chuẩn
#define GAS_CONSTANT    2000.0f     // Hằng số khí (điều chỉnh áp suất)
#define H               0.045f      // Bán kính làm mịn (càng lớn càng mượt)
#define H2              (H * H)     // H bình phương - tính sẵn cho nhanh
#define MASS            0.02f       // Khối lượng mỗi hạt
#define VISCOSITY       400.0f      // Độ nhớt (cao = nước chảy chậm hơn)
#define SURFACE_TENSION 0.0728f     // Sức căng bề mặt
#define DT              0.0003f     // Bước thời gian (nhỏ = ổn định hơn)
#define GRAVITY         -9.81f      // Gia tốc trọng trường

// Kích thước hộp chứa nước
#define BOX_WIDTH       1.875f
#define BOX_HEIGHT      1.25f
#define BOX_DEPTH       0.375f
#define BOUNDARY_DAMPING 0.3f       // Giảm vận tốc khi chạm tường

// Cấu hình CUDA - số threads mỗi block
#define BLOCK_SIZE      256

// Cửa sổ hiển thị
#define WINDOW_WIDTH    1400
#define WINDOW_HEIGHT   900
#define RENDER_SCALE    600.0f

// Các hằng số cho SPH kernel
#define PI 3.14159265359f
#define POLY6_CONST     (315.0f / (64.0f * PI * powf(H, 9)))
#define SPIKY_CONST     (-45.0f / (PI * powf(H, 6)))
#define VISC_LAP_CONST  (45.0f / (PI * powf(H, 6)))

// Tile size cho shared memory
#define TILE_SIZE       128

// ============================================================================
// CONSTANT MEMORY - Bộ nhớ constant trên GPU
// ============================================================================
// Dùng để lưu các hằng số, tất cả thread đều đọc được nhanh

__constant__ float d_H;              // Độ dài làm mịn
__constant__ float d_H2;             // H bình phương
__constant__ float d_MASS;           // Khối lượng hạt
__constant__ float d_REST_DENSITY;   // Mật độ chuẩn
__constant__ float d_GAS_CONSTANT;   // Hằng số khí
__constant__ float d_VISCOSITY;      // Độ nhớt
__constant__ float d_GRAVITY;        // Gia tốc trọng trường
__constant__ float d_POLY6;          // Hằng số kernel Poly6
__constant__ float d_SPIKY;          // Hằng số kernel Spiky
__constant__ float d_VISC_LAP;       // Hằng số Laplacian độ nhớt
__constant__ float d_BOX_WIDTH;      // Kích thước hộp chứa
__constant__ float d_BOX_HEIGHT;
__constant__ float d_BOX_DEPTH;
__constant__ float d_DAMPING;        // Hệ số giảm vận tốc khi va chạm
__constant__ float d_DT;             // Bước thời gian

// ============================================================================
// CẤU TRÚC DỮ LIỆU
// ============================================================================

// Cấu trúc lưu trữ dữ liệu các hạt nước
typedef struct {
    float* __restrict__ x;           // Vị trí X
    float* __restrict__ y;           // Vị trí Y
    float* __restrict__ z;           // Vị trí Z
    float* __restrict__ vx;          // Vận tốc X
    float* __restrict__ vy;          // Vận tốc Y
    float* __restrict__ vz;          // Vận tốc Z
    float* __restrict__ density;     // Mật độ
    float* __restrict__ pressure;    // Áp suất
    float* __restrict__ fx;          // Lực X
    float* __restrict__ fy;          // Lực Y
    float* __restrict__ fz;          // Lực Z
} Particles;



// ============================================================================
// QUẢN LÝ BỘ NHỚ
// ============================================================================

// Cấp phát memory cho particles trên CPU
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

// Giải phóng memory trên CPU
void freeParticlesHost(Particles* p) {
    free(p->x);  free(p->y);  free(p->z);
    free(p->vx); free(p->vy); free(p->vz);
    free(p->density); free(p->pressure);
    free(p->fx); free(p->fy); free(p->fz);
}

// Cấp phát memory trên GPU
void allocateParticlesDevice(Particles* p, int n) {
    cudaMalloc(&p->x,        n * sizeof(float));
    cudaMalloc(&p->y,        n * sizeof(float));
    cudaMalloc(&p->z,        n * sizeof(float));
    cudaMalloc(&p->vx,       n * sizeof(float));
    cudaMalloc(&p->vy,       n * sizeof(float));
    cudaMalloc(&p->vz,       n * sizeof(float));
    cudaMalloc(&p->density,  n * sizeof(float));
    cudaMalloc(&p->pressure, n * sizeof(float));
    cudaMalloc(&p->fx,       n * sizeof(float));
    cudaMalloc(&p->fy,       n * sizeof(float));
    cudaMalloc(&p->fz,       n * sizeof(float));
}

// Giải phóng memory trên GPU
void freeParticlesDevice(Particles* p) {
    cudaFree(p->x);  cudaFree(p->y);  cudaFree(p->z);
    cudaFree(p->vx); cudaFree(p->vy); cudaFree(p->vz);
    cudaFree(p->density); cudaFree(p->pressure);
    cudaFree(p->fx); cudaFree(p->fy); cudaFree(p->fz);
}

// Copy dữ liệu từ CPU lên GPU
void copyParticlesToDevice(Particles* dst, Particles* src, int n) {
    cudaMemcpy(dst->x,  src->x,  n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dst->y,  src->y,  n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dst->z,  src->z,  n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dst->vx, src->vx, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dst->vy, src->vy, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dst->vz, src->vz, n * sizeof(float), cudaMemcpyHostToDevice);
}

// Copy dữ liệu từ GPU xuống CPU
void copyParticlesToHost(Particles* dst, Particles* src, int n) {
    cudaMemcpy(dst->x,  src->x,  n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dst->y,  src->y,  n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dst->z,  src->z,  n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dst->vx, src->vx, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dst->vy, src->vy, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dst->vz, src->vz, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dst->density, src->density, n * sizeof(float), cudaMemcpyDeviceToHost);
}

// ============================================================================
// KHỚI TẠO CÁC HẠT NƯỚC
// ============================================================================

// Khởi tạo các hạt nước - Dam break
void initWaterBlock(Particles* p, int* numParticles) {
    int count = 0;
    float spacing = H * 0.5f;
    
    for (float x = -BOX_WIDTH/2 + spacing; x < BOX_WIDTH/2 - spacing && count < INITIAL_PARTICLES; x += spacing) {
        for (float y = -BOX_HEIGHT/2 + spacing; y < BOX_HEIGHT/2 - spacing && count < INITIAL_PARTICLES; y += spacing) {
            for (float z = -BOX_DEPTH/2 + spacing; z < BOX_DEPTH/2 - spacing && count < INITIAL_PARTICLES; z += spacing) {
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
    
    *numParticles = count;
}

// ============================================================================
// CÁC HÀM CUDA KERNEL - CHẠY TRÊN GPU
// ============================================================================

// Hàm Poly6 Kernel - dùng cho tính mật độ
__device__ __forceinline__ float poly6Kernel(float r2) {
    if (r2 >= d_H2) return 0.0f;
    float diff = d_H2 - r2;
    return d_POLY6 * diff * diff * diff;
}

// Hàm Spiky Kernel - dùng cho lực áp suất
__device__ __forceinline__ float spikyKernelGrad(float r) {
    if (r >= d_H || r < 1e-6f) return 0.0f;
    float diff = d_H - r;
    return d_SPIKY * diff * diff;
}

// Hàm Viscosity Kernel - dùng cho lực ma sát
__device__ __forceinline__ float viscosityKernelLap(float r) {
    if (r >= d_H) return 0.0f;
    return d_VISC_LAP * (d_H - r);
}

/*
 * KERNEL 1: Tính mật độ và áp suất
 * Mỗi thread tính cho 1 particle
 */
__global__ void computeDensityPressureKernel(
    const float* __restrict__ x, 
    const float* __restrict__ y, 
    const float* __restrict__ z,
    float* __restrict__ density, 
    float* __restrict__ pressure,
    int n
) {
    // Shared memory - cache vị trí particles trong tile
    __shared__ float tile_x[TILE_SIZE];
    __shared__ float tile_y[TILE_SIZE];
    __shared__ float tile_z[TILE_SIZE];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load vị trí particle hiện tại vào register
    float xi, yi, zi;
    if (i < n) {
        xi = x[i];
        yi = y[i];
        zi = z[i];
    }
    
    float rho = 0.0f;  // Biến lưu mật độ
    
    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;  // Số tile cần duyệt
    
    // Duyệt qua từng tile
    for (int tile = 0; tile < numTiles; tile++) {
        int loadIdx = tile * TILE_SIZE + threadIdx.x;
        
        // Mỗi thread load 1 particle vào shared memory
        if (loadIdx < n && threadIdx.x < TILE_SIZE) {
            tile_x[threadIdx.x] = x[loadIdx];
            tile_y[threadIdx.x] = y[loadIdx];
            tile_z[threadIdx.x] = z[loadIdx];
        }
        
        // Đợi tất cả threads load xong
        __syncthreads();
        
        // Tính mật độ từ particles trong tile này
        if (i < n) {
            int tileEnd = min(TILE_SIZE, n - tile * TILE_SIZE);
            
            // Loop unrolling - compiler tự tối ưu
            #pragma unroll 8
            for (int j = 0; j < tileEnd; j++) {
                float dx = xi - tile_x[j];
                float dy = yi - tile_y[j];
                float dz = zi - tile_z[j];
                
                // Tính khoảng cách bình phương
                float r2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
                
                rho += d_MASS * poly6Kernel(r2);
            }
        }
        
        // Đợi tất cả threads tính xong trước khi load tile mới
        __syncthreads();
    }
    
    if (i < n) {
        density[i] = rho;
        
        // Tính áp suất từ mật độ (công thức Tait)
        float p = d_GAS_CONSTANT * (rho - d_REST_DENSITY);
        
        // Áp suất không thể âm
        pressure[i] = fmaxf(p, 0.0f);
    }
}

/*
 * KERNEL 2: Tính các lực tác dụng lên mỗi hạt
 * Bao gồm: Lực áp suất, lực ma sát
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
    // Shared memory - cache particle data
    __shared__ float tile_x[TILE_SIZE];
    __shared__ float tile_y[TILE_SIZE];
    __shared__ float tile_z[TILE_SIZE];
    __shared__ float tile_vx[TILE_SIZE];
    __shared__ float tile_vy[TILE_SIZE];
    __shared__ float tile_vz[TILE_SIZE];
    __shared__ float tile_density[TILE_SIZE];
    __shared__ float tile_pressure[TILE_SIZE];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load particle hiện tại vào register
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
    
    // Biến lưu lực
    float forceX = 0.0f;
    float forceY = 0.0f;
    float forceZ = 0.0f;
    
    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < numTiles; tile++) {
        int loadIdx = tile * TILE_SIZE + threadIdx.x;
        
        // Load tile vào shared memory
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
                // Bỏ qua particle chính nó
                if (globalJ + j == i) continue;
                
                float dx = xi - tile_x[j];
                float dy = yi - tile_y[j];
                float dz = zi - tile_z[j];
                float r2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
                
                // Nếu quá xa thì bỏ qua
                if (r2 >= d_H2 || r2 < 1e-12f) continue;
                
                // Tính khoảng cách (dùng rsqrtf cho nhanh)
                float r = rsqrtf(r2);
                r = 1.0f / r;
                
                float rhoj = tile_density[j];
                float pj = tile_pressure[j];
                
                // Tính reciprocal 1 lần, dùng nhiều lần
                float inv_rhoj = 1.0f / rhoj;
                float inv_r = 1.0f / r;
                
                // Lực áp suất
                float pressureForce = -d_MASS * (pi + pj) * 0.5f * inv_rhoj * spikyKernelGrad(r);
                
                // Dùng fmaf cho multiply-add nhanh hơn
                forceX = fmaf(pressureForce * inv_r, dx, forceX);
                forceY = fmaf(pressureForce * inv_r, dy, forceY);
                forceZ = fmaf(pressureForce * inv_r, dz, forceZ);
                
                // Lực ma sát (viscosity)
                float viscForce = d_VISCOSITY * d_MASS * inv_rhoj * viscosityKernelLap(r);
                forceX = fmaf(viscForce, tile_vx[j] - vxi, forceX);
                forceY = fmaf(viscForce, tile_vy[j] - vyi, forceY);
                forceZ = fmaf(viscForce, tile_vz[j] - vzi, forceZ);
            }
        }
        
        __syncthreads();
    }
    
    if (i < n) {
        // Thêm trọng lực
        forceY = fmaf(rhoi, d_GRAVITY, forceY);
        
        fx[i] = forceX;
        fy[i] = forceY;
        fz[i] = forceZ;
    }
}

/*
 * KERNEL 3: Tích phân - Cập nhật vị trí và vận tốc
 * Tính gia tốc từ lực, rồi cập nhật vận tốc và vị trí
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
    
    // Load mật độ và check an toàn
    float rhoi = density[i];
    rhoi = fmaxf(rhoi, 1e-6f);  // Tránh chia cho 0
    
    float inv_rho = 1.0f / rhoi;
    
    // Tính gia tốc = Lực / Mật độ
    float ax = fx[i] * inv_rho;
    float ay = fy[i] * inv_rho;
    float az = fz[i] * inv_rho;
    
    // Giới hạn gia tốc (không quá lớn)
    float accMag2 = fmaf(ax, ax, fmaf(ay, ay, az * az));
    const float maxAcc = 500.0f;
    const float maxAcc2 = maxAcc * maxAcc;
    
    if (accMag2 > maxAcc2) {
        float scale = maxAcc * rsqrtf(accMag2);
        ax *= scale;
        ay *= scale;
        az *= scale;
    }
    
    // Load vận tốc hiện tại
    float vxi = vx[i];
    float vyi = vy[i];
    float vzi = vz[i];
    
    // Cập nhật vận tốc: v = v + a*dt
    vxi = fmaf(ax, d_DT, vxi);
    vyi = fmaf(ay, d_DT, vyi);
    vzi = fmaf(az, d_DT, vzi);
    
    // Load vị trí hiện tại
    float xi = x[i];
    float yi = y[i];
    float zi = z[i];
    
    // Cập nhật vị trí: x = x + v*dt
    xi = fmaf(vxi, d_DT, xi);
    yi = fmaf(vyi, d_DT, yi);
    zi = fmaf(vzi, d_DT, zi);
    
    // Xử lý va chạm với tường
    float halfWidth = d_BOX_WIDTH * 0.5f;
    float halfHeight = d_BOX_HEIGHT * 0.5f;
    float halfDepth = d_BOX_DEPTH * 0.5f;
    
    // Check biên X
    if (xi < -halfWidth) {
        xi = -halfWidth;
        vxi *= -d_DAMPING;
    } else if (xi > halfWidth) {
        xi = halfWidth;
        vxi *= -d_DAMPING;
    }
    
    // Check biên Y
    if (yi < -halfHeight) {
        yi = -halfHeight;
        vyi *= -d_DAMPING;
    } else if (yi > halfHeight) {
        yi = halfHeight;
        vyi *= -d_DAMPING;
    }
    
    // Check biên Z
    if (zi < -halfDepth) {
        zi = -halfDepth;
        vzi *= -d_DAMPING;
    } else if (zi > halfDepth) {
        zi = halfDepth;
        vzi *= -d_DAMPING;
    }
    
    // Ghi lại vào memory
    x[i] = xi;
    y[i] = yi;
    z[i] = zi;
    vx[i] = vxi;
    vy[i] = vyi;
    vz[i] = vzi;
}

// ============================================================================
// PHẦN HIỂN THỊ - Vẽ lên màn hình
// ============================================================================

// Vẽ nước và hộp chứa
void renderWater(SDL_Renderer* renderer, Particles* p, int n) {
    // Vẽ viền hộp
    float scale = RENDER_SCALE;
    int boxLeft   = (int)((-BOX_WIDTH/2) * scale + WINDOW_WIDTH/2);
    int boxRight  = (int)((BOX_WIDTH/2) * scale + WINDOW_WIDTH/2);
    int boxTop    = (int)((-BOX_HEIGHT/2) * scale + WINDOW_HEIGHT/2);
    int boxBottom = (int)((BOX_HEIGHT/2) * scale + WINDOW_HEIGHT/2);
    
    // Lật trục Y cho toạ độ màn hình
    int temp = boxTop;
    boxTop = WINDOW_HEIGHT - boxBottom;
    boxBottom = WINDOW_HEIGHT - temp;
    
    // Vẽ 4 cạnh hộp
    
    SDL_SetRenderDrawColor(renderer, 100, 100, 100, 255);
    SDL_RenderLine(renderer, (float)boxLeft, (float)boxTop, (float)boxRight, (float)boxTop);
    SDL_RenderLine(renderer, (float)boxRight, (float)boxTop, (float)boxRight, (float)boxBottom);
    SDL_RenderLine(renderer, (float)boxRight, (float)boxBottom, (float)boxLeft, (float)boxBottom);
    SDL_RenderLine(renderer, (float)boxLeft, (float)boxBottom, (float)boxLeft, (float)boxTop);
    
    // Tìm vận tốc max để đổi màu
    float maxVel = 0.1f;
    for (int i = 0; i < n; i++) {
        float vel = sqrtf(p->vx[i]*p->vx[i] + p->vy[i]*p->vy[i] + p->vz[i]*p->vz[i]);
        if (vel > maxVel) maxVel = vel;
    }
    
    // Vẽ từng hạt nước
    for (int i = 0; i < n; i++) {
        int sx = (int)(p->x[i] * scale + WINDOW_WIDTH/2);
        int sy = WINDOW_HEIGHT - (int)(p->y[i] * scale + WINDOW_HEIGHT/2);
        
        // Bỏ qua nếu ra ngoài màn hình
        if (sx < 0 || sx >= WINDOW_WIDTH || sy < 0 || sy >= WINDOW_HEIGHT) continue;
        
        // Màu sắc dựa vào vận tốc (xanh dương -> trắng khi chảy nhanh)
        float vel = sqrtf(p->vx[i]*p->vx[i] + p->vy[i]*p->vy[i] + p->vz[i]*p->vz[i]);
        float velNorm = vel / maxVel;
        
        // Mật độ ảnh hưởng độ đậm
        float densityNorm = p->density[i] / (REST_DENSITY * 1.5f);
        if (densityNorm > 1.0f) densityNorm = 1.0f;
        
        // Tạo màu xanh dương cho nước
        int r, g, b;
        if (velNorm < 0.3f) {
            // Nước đứng yên - xanh đậm
            r = 20 + (int)(30 * densityNorm);
            g = 80 + (int)(50 * densityNorm);
            b = 180 + (int)(50 * densityNorm);
        } else if (velNorm < 0.6f) {
            // Nước đang chảy - xanh nhạt hơn
            float t = (velNorm - 0.3f) / 0.3f;
            r = 50 + (int)(100 * t);
            g = 130 + (int)(70 * t);
            b = 200 + (int)(30 * t);
        } else {
            // Chảy nhanh / bắn tản - trắng xám
            float t = (velNorm - 0.6f) / 0.4f;
            r = 150 + (int)(105 * t);
            g = 200 + (int)(55 * t);
            b = 230 + (int)(25 * t);
        }
        
        SDL_SetRenderDrawColor(renderer, r, g, b, 255);
        
        // Vẽ hạt như hình tròn nhỏ (2 pixels)
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

// Vẽ UI - thanh FPS
void renderUI(SDL_Renderer* renderer, float fps) {
    float fpsNorm = fps / 60.0f;
    if (fpsNorm > 1.0f) fpsNorm = 1.0f;
    SDL_SetRenderDrawColor(renderer, 100, 255, 100, 200);
    SDL_FRect fpsBar = {10, 10, fpsNorm * 100, 10};
    SDL_RenderFillRect(renderer, &fpsBar);
}

// ============================================================================
// HÀM MAIN - CHƯƠNG TRÌNH CHÍNH
// ============================================================================

// Khởi tạo constant memory trên GPU
// Copy các hằng số lên GPU để truy cập nhanh hơn
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
    
    cudaMemcpyToSymbol(d_H, &h, sizeof(float));
    cudaMemcpyToSymbol(d_H2, &h2, sizeof(float));
    cudaMemcpyToSymbol(d_MASS, &mass, sizeof(float));
    cudaMemcpyToSymbol(d_REST_DENSITY, &restDensity, sizeof(float));
    cudaMemcpyToSymbol(d_GAS_CONSTANT, &gasConstant, sizeof(float));
    cudaMemcpyToSymbol(d_VISCOSITY, &viscosity, sizeof(float));
    cudaMemcpyToSymbol(d_GRAVITY, &gravity, sizeof(float));
    cudaMemcpyToSymbol(d_POLY6, &poly6, sizeof(float));
    cudaMemcpyToSymbol(d_SPIKY, &spiky, sizeof(float));
    cudaMemcpyToSymbol(d_VISC_LAP, &viscLap, sizeof(float));
    cudaMemcpyToSymbol(d_BOX_WIDTH, &boxWidth, sizeof(float));
    cudaMemcpyToSymbol(d_BOX_HEIGHT, &boxHeight, sizeof(float));
    cudaMemcpyToSymbol(d_BOX_DEPTH, &boxDepth, sizeof(float));
    cudaMemcpyToSymbol(d_DAMPING, &damping, sizeof(float));
    cudaMemcpyToSymbol(d_DT, &dt, sizeof(float));
}

int main(int argc, char* argv[]) {
    // Khoi tao SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("Loi SDL: %s\n", SDL_GetError());
        return 1;
    }
    
    SDL_Window* window = SDL_CreateWindow(
        "Mo phong nuoc CUDA - SPH",
        WINDOW_WIDTH, WINDOW_HEIGHT,
        0
    );
    
    if (!window) {
        printf("Loi tao window: %s\n", SDL_GetError());
        return 1;
    }
    
    SDL_Renderer* renderer = SDL_CreateRenderer(window, NULL);
    if (!renderer) {
        printf("Loi tao renderer: %s\n", SDL_GetError());
        return 1;
    }
    
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    
    initConstantMemory();
    
    // Cap phat bo nho
    Particles hostParticles, deviceParticles;
    allocateParticlesHost(&hostParticles, MAX_PARTICLES);
    allocateParticlesDevice(&deviceParticles, MAX_PARTICLES);
    
    // Tao cac hat nuoc ban dau
    srand((unsigned int)time(NULL));
    int numParticles = 0;
    initWaterBlock(&hostParticles, &numParticles);
    copyParticlesToDevice(&deviceParticles, &hostParticles, numParticles);
    
    int numBlocks = (numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Timing
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    
    // Vong lap chinh
    bool running = true;
    SDL_Event event;
    
    Uint64 frameStart = SDL_GetPerformanceCounter();
    Uint64 freq = SDL_GetPerformanceFrequency();
    int frameCount = 0;
    float fps = 60.0f;
    float totalSimTime = 0.0f;
    
    int subSteps = 6;
    
    while (running) {
        // Xu ly su kien
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) {
                running = false;
            } else if (event.type == SDL_EVENT_KEY_DOWN) {
                if (event.key.key == SDLK_ESCAPE) {
                    running = false;
                }
            } else if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
                if (event.button.button == SDL_BUTTON_LEFT && numParticles < MAX_PARTICLES - 100) {
                    float scale = RENDER_SCALE;
                    float clickX = (event.button.x - WINDOW_WIDTH/2) / scale;
                    float clickY = -(event.button.y - WINDOW_HEIGHT/2) / scale;
                    
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
                }
            }
        }
        
        // Cap nhat physics
        cudaEventRecord(startEvent);
            
            for (int step = 0; step < subSteps; step++) {
                // Buoc 1: Tinh mat do va ap suat
                computeDensityPressureKernel<<<numBlocks, BLOCK_SIZE>>>(
                    deviceParticles.x, deviceParticles.y, deviceParticles.z,
                    deviceParticles.density, deviceParticles.pressure,
                    numParticles
                );
                
                // Buoc 2: Tinh cac luc
                computeForcesKernel<<<numBlocks, BLOCK_SIZE>>>(
                    deviceParticles.x, deviceParticles.y, deviceParticles.z,
                    deviceParticles.vx, deviceParticles.vy, deviceParticles.vz,
                    deviceParticles.density, deviceParticles.pressure,
                    deviceParticles.fx, deviceParticles.fy, deviceParticles.fz,
                    numParticles
                );
                
                // Buoc 3: Cap nhat vi tri va van toc
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
        
        // Copy ket qua ve CPU
        copyParticlesToHost(&hostParticles, &deviceParticles, numParticles);
        
        // Ve len man hinh
        SDL_SetRenderDrawColor(renderer, 10, 15, 30, 255);
        SDL_RenderClear(renderer);
        
        renderWater(renderer, &hostParticles, numParticles);
        renderUI(renderer, fps);
        
        SDL_RenderPresent(renderer);
        
        // Tinh FPS
        frameCount++;
        Uint64 now = SDL_GetPerformanceCounter();
        float elapsed = (float)(now - frameStart) / freq;
        if (elapsed >= 1.0f) {
            fps = frameCount / elapsed;
            frameCount = 0;
            frameStart = now;
            
            // Cap nhat tieu de cua so
            char title[256];
            sprintf(title, "Mo phong nuoc CUDA | Hat: %d | FPS: %.1f | Thoi gian: %.2fs", 
                    numParticles, fps, totalSimTime);
            SDL_SetWindowTitle(window, title);
        }
    }
    
    // Don dep
    printf("\n\n=== KET THUC ===\n");
    printf("Thoi gian mo phong: %.2f giay\n", totalSimTime);
    printf("So hat: %d\n", numParticles);
    
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    
    freeParticlesHost(&hostParticles);
    freeParticlesDevice(&deviceParticles);
    
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    
    return 0;
}
