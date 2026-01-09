/**
 * ═══════════════════════════════════════════════════════════════════════════════
 *                    CUDA N-BODY SIMULATION (BRUTE FORCE O(N²))
 * ═══════════════════════════════════════════════════════════════════════════════
 * 
 * Author: N-Body Paralleling Project
 * Description: GPU-accelerated N-body gravitational simulation using CUDA
 *              with SDL3 real-time visualization
 * 
 * Algorithm: Direct summation (brute force) with tiled shared memory optimization
 * Complexity: O(N²) per time step
 * 
 * Features:
 *   - Structure of Arrays (SoA) memory layout for coalesced GPU access
 *   - Shared memory tiling to reduce global memory bandwidth
 *   - Loop unrolling for instruction-level parallelism
 *   - Fast math intrinsics (rsqrtf)
 *   - Beautiful galaxy visualization with color-coded velocities
 *   - Interactive camera controls
 * 
 * Controls:
 *   ESC      - Exit and show performance report
 *   SPACE    - Pause/Resume simulation  
 *   R        - Reset simulation
 *   +/-      - Increase/Decrease zoom
 *   Arrow    - Pan camera
 * 
 * Performance (RTX A1000, N=50k): ~15ms/step
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
// Recommended: 10k-100k for O(N²) brute force
// Beyond 100k, consider Barnes-Hut or grid-based algorithms
#define N 10000

// --- Physics Constants ---
#define DT          0.005f      // Time step (smaller = more accurate, slower)
#define G           1.0f        // Gravitational constant
#define SOFTENING   0.1f        // Softening factor to avoid singularities

// --- CUDA Kernel Configuration ---
#define BLOCK_SIZE  256         // Threads per block (256 is optimal for most GPUs)
#define UNROLL_FACTOR 32        // Loop unrolling factor

// --- Visualization ---
#define WINDOW_WIDTH  1200
#define WINDOW_HEIGHT 900
#define INITIAL_SCALE 3.5f      // Initial zoom level

// --- Initial Conditions Mode (press 1-4 to change at runtime) ---
// 1: Single galaxy (spiral)
// 2: Two colliding galaxies  
// 3: Random distribution
// 4: Ring formation
#define INIT_MODE 1

// Global variable for runtime mode switching
int currentMode = INIT_MODE;

// ═══════════════════════════════════════════════════════════════════════════════
//                              DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Structure of Arrays (SoA) layout for particles
 * This layout ensures coalesced memory access on GPU
 * Much faster than Array of Structures (AoS) for CUDA
 */
typedef struct {
    float* x;       // Position X
    float* y;       // Position Y  
    float* z;       // Position Z
    float* vx;      // Velocity X
    float* vy;      // Velocity Y
    float* vz;      // Velocity Z
    float* mass;    // Mass
} Bodies;

// Camera state for interactive visualization
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

void allocateBodiesHost(Bodies* b, int n) {
    b->x    = (float*)malloc(n * sizeof(float));
    b->y    = (float*)malloc(n * sizeof(float));
    b->z    = (float*)malloc(n * sizeof(float));
    b->vx   = (float*)malloc(n * sizeof(float));
    b->vy   = (float*)malloc(n * sizeof(float));
    b->vz   = (float*)malloc(n * sizeof(float));
    b->mass = (float*)malloc(n * sizeof(float));
}

void freeBodiesHost(Bodies* b) {
    free(b->x);  free(b->y);  free(b->z);
    free(b->vx); free(b->vy); free(b->vz);
    free(b->mass);
}

void allocateBodiesDevice(Bodies* b, int n) {
    CUDA_CHECK(cudaMalloc(&b->x,    n * sizeof(float)), "malloc x");
    CUDA_CHECK(cudaMalloc(&b->y,    n * sizeof(float)), "malloc y");
    CUDA_CHECK(cudaMalloc(&b->z,    n * sizeof(float)), "malloc z");
    CUDA_CHECK(cudaMalloc(&b->vx,   n * sizeof(float)), "malloc vx");
    CUDA_CHECK(cudaMalloc(&b->vy,   n * sizeof(float)), "malloc vy");
    CUDA_CHECK(cudaMalloc(&b->vz,   n * sizeof(float)), "malloc vz");
    CUDA_CHECK(cudaMalloc(&b->mass, n * sizeof(float)), "malloc mass");
}

void freeBodiesDevice(Bodies* b) {
    cudaFree(b->x);  cudaFree(b->y);  cudaFree(b->z);
    cudaFree(b->vx); cudaFree(b->vy); cudaFree(b->vz);
    cudaFree(b->mass);
}

void copyBodiesToDevice(Bodies* dst, Bodies* src, int n) {
    CUDA_CHECK(cudaMemcpy(dst->x,    src->x,    n * sizeof(float), cudaMemcpyHostToDevice), "H2D x");
    CUDA_CHECK(cudaMemcpy(dst->y,    src->y,    n * sizeof(float), cudaMemcpyHostToDevice), "H2D y");
    CUDA_CHECK(cudaMemcpy(dst->z,    src->z,    n * sizeof(float), cudaMemcpyHostToDevice), "H2D z");
    CUDA_CHECK(cudaMemcpy(dst->vx,   src->vx,   n * sizeof(float), cudaMemcpyHostToDevice), "H2D vx");
    CUDA_CHECK(cudaMemcpy(dst->vy,   src->vy,   n * sizeof(float), cudaMemcpyHostToDevice), "H2D vy");
    CUDA_CHECK(cudaMemcpy(dst->vz,   src->vz,   n * sizeof(float), cudaMemcpyHostToDevice), "H2D vz");
    CUDA_CHECK(cudaMemcpy(dst->mass, src->mass, n * sizeof(float), cudaMemcpyHostToDevice), "H2D mass");
}

void copyPositionsToHost(Bodies* dst, Bodies* src, int n) {
    CUDA_CHECK(cudaMemcpy(dst->x, src->x, n * sizeof(float), cudaMemcpyDeviceToHost), "D2H x");
    CUDA_CHECK(cudaMemcpy(dst->y, src->y, n * sizeof(float), cudaMemcpyDeviceToHost), "D2H y");
    CUDA_CHECK(cudaMemcpy(dst->z, src->z, n * sizeof(float), cudaMemcpyDeviceToHost), "D2H z");
}

void copyVelocitiesToHost(Bodies* dst, Bodies* src, int n) {
    CUDA_CHECK(cudaMemcpy(dst->vx, src->vx, n * sizeof(float), cudaMemcpyDeviceToHost), "D2H vx");
    CUDA_CHECK(cudaMemcpy(dst->vy, src->vy, n * sizeof(float), cudaMemcpyDeviceToHost), "D2H vy");
}

// ═══════════════════════════════════════════════════════════════════════════════
//                         INITIAL CONDITIONS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * MODE 1: Single spiral galaxy
 * Beautiful spiral with central supermassive black hole
 */
void initSingleGalaxy(Bodies* p, int n) {
    printf("  -> Initializing: SPIRAL GALAXY\\n");
    
    // Central supermassive black hole
    p->x[0] = 0; p->y[0] = 0; p->z[0] = 0;
    p->vx[0] = 0; p->vy[0] = 0; p->vz[0] = 0;
    p->mass[0] = 10000.0f;  // Very massive central body
    
    // Create beautiful spiral arms
    int numArms = 2;  // 2 main spiral arms (like Milky Way)
    float armWidth = 0.4f;  // Width of each arm
    
    for (int i = 1; i < n; ++i) {
        int arm = i % numArms;
        float armAngle = arm * 3.14159f;  // 180 degrees apart
        
        // Logarithmic spiral
        float t = (float)i / n;
        float spiralAngle = t * 4.0f * 3.14159f;  // ~2 full rotations
        float baseRadius = 10.0f + t * 90.0f;
        
        // Tighter arms near center, spread out further away
        float spreadFactor = armWidth * (0.3f + t * 0.7f);
        float angleSpread = ((float)(rand() % 1000) / 1000.0f - 0.5f) * spreadFactor;
        float radiusSpread = ((float)(rand() % 1000) / 1000.0f - 0.5f) * baseRadius * 0.15f;
        
        float finalAngle = spiralAngle + armAngle + angleSpread;
        float finalRadius = baseRadius + radiusSpread;
        
        p->x[i] = cosf(finalAngle) * finalRadius;
        p->y[i] = sinf(finalAngle) * finalRadius;
        p->z[i] = ((float)(rand() % 100) / 100.0f - 0.5f) * (5.0f * (1.0f - t * 0.8f));
        
        // Keplerian orbital velocity
        float orbitalVel = sqrtf(G * 10000.0f / finalRadius) * 0.95f;
        p->vx[i] = -sinf(finalAngle) * orbitalVel;
        p->vy[i] = cosf(finalAngle) * orbitalVel;
        p->vz[i] = 0;
        
        // Varied star masses
        p->mass[i] = 0.3f + ((float)(rand() % 100) / 100.0f) * 1.2f;
    }
}

/**
 * MODE 2: Two colliding galaxies
 * Dramatic collision of two spiral galaxies
 */
void initCollidingGalaxies(Bodies* p, int n) {
    printf("  -> Initializing: GALAXY COLLISION\n");
    
    float separation = 120.0f;  // Distance between galaxy centers
    
    // GALAXY 1 (Blue - Left side, moving right)
    int n1 = n / 2;
    p->x[0] = -separation/2;
    p->y[0] = 0.0f;
    p->z[0] = 0.0f;
    p->vx[0] = 1.2f;   // Moving right
    p->vy[0] = 0.3f;   // Slight upward
    p->vz[0] = 0.0f;
    p->mass[0] = 5000.0f;
    
    for (int i = 1; i < n1; ++i) {
        float t = (float)i / n1;
        float angle = t * 4.0f * 3.14159f + (i % 2) * 3.14159f;
        float dist = 5.0f + t * 50.0f + ((float)(rand() % 100) / 100.0f - 0.5f) * 8.0f;
        
        p->x[i] = -separation/2 + cosf(angle) * dist;
        p->y[i] = sinf(angle) * dist;
        p->z[i] = ((float)(rand() % 100) / 100.0f - 0.5f) * 4.0f;
        
        float orbitalVel = sqrtf(G * 5000.0f / dist) * 0.95f;
        p->vx[i] = 1.2f - sinf(angle) * orbitalVel;
        p->vy[i] = 0.3f + cosf(angle) * orbitalVel;
        p->vz[i] = 0.0f;
        p->mass[i] = 0.5f + ((float)(rand() % 100) / 100.0f);
    }
    
    // GALAXY 2 (Red - Right side, moving left)
    p->x[n1] = separation/2;
    p->y[n1] = 30.0f;  // Offset vertically for oblique collision
    p->z[n1] = 0.0f;
    p->vx[n1] = -1.0f;  // Moving left
    p->vy[n1] = -0.4f;  // Moving down
    p->vz[n1] = 0.0f;
    p->mass[n1] = 4000.0f;
    
    for (int i = n1 + 1; i < n; ++i) {
        int j = i - n1;
        float t = (float)j / (n - n1);
        float angle = t * 4.0f * 3.14159f + (j % 2) * 3.14159f + 0.5f;
        float dist = 5.0f + t * 45.0f + ((float)(rand() % 100) / 100.0f - 0.5f) * 8.0f;
        
        p->x[i] = separation/2 + cosf(angle) * dist;
        p->y[i] = 30.0f + sinf(angle) * dist;
        p->z[i] = ((float)(rand() % 100) / 100.0f - 0.5f) * 4.0f;
        
        float orbitalVel = sqrtf(G * 4000.0f / dist) * 0.95f;
        p->vx[i] = -1.0f - sinf(angle) * orbitalVel;
        p->vy[i] = -0.4f + cosf(angle) * orbitalVel;
        p->vz[i] = 0.0f;
        p->mass[i] = 0.5f + ((float)(rand() % 100) / 100.0f);
    }
}

/**
 * MODE 3: Chaotic random cluster
 * Random distribution with multiple massive bodies - forms chaotic structures
 */
void initRandom(Bodies* p, int n) {
    printf("  -> Initializing: RANDOM CHAOS\n");
    
    // Create several massive "seeds" that will attract particles
    int numSeeds = 8;
    for (int i = 0; i < numSeeds; ++i) {
        float angle = (float)i / numSeeds * 2.0f * 3.14159f;
        float dist = 40.0f + (rand() % 40);
        p->x[i] = cosf(angle) * dist;
        p->y[i] = sinf(angle) * dist;
        p->z[i] = ((float)(rand() % 100) / 100.0f - 0.5f) * 20.0f;
        p->vx[i] = ((float)(rand() % 100) / 100.0f - 0.5f) * 0.8f;
        p->vy[i] = ((float)(rand() % 100) / 100.0f - 0.5f) * 0.8f;
        p->vz[i] = 0;
        p->mass[i] = 800.0f + (rand() % 800);  // Massive seeds
    }
    
    // Random particles spread around
    for (int i = numSeeds; i < n; ++i) {
        // Spherical distribution
        float theta = ((float)(rand() % 10000) / 10000.0f) * 2.0f * 3.14159f;
        float phi = acosf(2.0f * ((float)(rand() % 10000) / 10000.0f) - 1.0f);
        float r = 20.0f + ((float)(rand() % 10000) / 10000.0f) * 100.0f;
        
        p->x[i] = r * sinf(phi) * cosf(theta);
        p->y[i] = r * sinf(phi) * sinf(theta);
        p->z[i] = r * cosf(phi) * 0.3f;  // Flatten z
        
        // Small random velocities
        p->vx[i] = ((float)(rand() % 100) / 100.0f - 0.5f) * 0.3f;
        p->vy[i] = ((float)(rand() % 100) / 100.0f - 0.5f) * 0.3f;
        p->vz[i] = 0;
        p->mass[i] = 0.5f + ((float)(rand() % 100) / 100.0f) * 1.5f;
    }
}

/**
 * MODE 4: Saturn-like ring system
 * Central body with beautiful concentric rings
 */
void initRing(Bodies* p, int n) {
    printf("  -> Initializing: RING SYSTEM (Saturn-like)\n");
    
    // Central massive body (like Saturn)
    p->x[0] = 0; p->y[0] = 0; p->z[0] = 0;
    p->vx[0] = 0; p->vy[0] = 0; p->vz[0] = 0;
    p->mass[0] = 10000.0f;  // Very massive central body
    
    // Define ring structure (like Saturn's rings: C, B, A rings with gaps)
    float ringRadii[] = {25.0f, 35.0f, 50.0f, 70.0f, 90.0f};  // Ring center radii
    float ringWidths[] = {8.0f, 12.0f, 10.0f, 15.0f, 12.0f};  // Ring widths
    int numRings = 5;
    
    int particlesPerRing = (n - 1) / numRings;
    int idx = 1;
    
    for (int ring = 0; ring < numRings && idx < n; ++ring) {
        float ringRadius = ringRadii[ring];
        float ringWidth = ringWidths[ring];
        int ringParticles = (ring == numRings - 1) ? (n - idx) : particlesPerRing;
        
        for (int i = 0; i < ringParticles && idx < n; ++i, ++idx) {
            float angle = ((float)i / ringParticles) * 2.0f * 3.14159f;
            angle += ((float)(rand() % 100) / 100.0f - 0.5f) * 0.1f;  // Small angular variation
            
            float dist = ringRadius + ((float)(rand() % 100) / 100.0f - 0.5f) * ringWidth;
            
            p->x[idx] = cosf(angle) * dist;
            p->y[idx] = sinf(angle) * dist;
            p->z[idx] = ((float)(rand() % 100) / 100.0f - 0.5f) * 1.5f;  // Very thin rings
            
            // Keplerian orbital velocity
            float orbitalVel = sqrtf(G * 10000.0f / dist);
            p->vx[idx] = -sinf(angle) * orbitalVel;
            p->vy[idx] = cosf(angle) * orbitalVel;
            p->vz[idx] = 0;
            p->mass[idx] = 0.3f + ((float)(rand() % 100) / 100.0f) * 0.4f;  // Small ring particles
        }
    }
}

const char* getModeNameByIndex(int mode) {
    switch (mode) {
        case 1: return "Spiral Galaxy";
        case 2: return "Galaxy Collision";
        case 3: return "Random Chaos";
        case 4: return "Ring System";
        default: return "Unknown";
    }
}

void initBodies(Bodies* p, int n, int mode) {
    printf("\nInitializing mode %d: %s\n", mode, getModeNameByIndex(mode));
    switch (mode) {
        case 1:
            initSingleGalaxy(p, n);
            break;
        case 2:
            initCollidingGalaxies(p, n);
            break;
        case 3:
            initRandom(p, n);
            break;
        case 4:
            initRing(p, n);
            break;
        default:
            initSingleGalaxy(p, n);
    }
    printf("  -> Done! %d particles created.\n\n", n);
}

// ═══════════════════════════════════════════════════════════════════════════════
//                              CUDA KERNELS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Gravitational force computation kernel using tiled shared memory
 * 
 * Optimization techniques:
 * 1. Shared memory tiling - reduces global memory access
 * 2. Loop unrolling - increases instruction-level parallelism
 * 3. Fast math (rsqrtf) - hardware-accelerated reciprocal square root
 * 4. Register blocking - position stored in registers
 * 5. __restrict__ - helps compiler optimize memory access
 * 6. __launch_bounds__ - hints for optimal register allocation
 */
__global__ void __launch_bounds__(BLOCK_SIZE, 4) 
bodyForceKernel(
    const float* __restrict__ px, 
    const float* __restrict__ py, 
    const float* __restrict__ pz,
    float* __restrict__ vx, 
    float* __restrict__ vy, 
    float* __restrict__ vz,
    const float* __restrict__ mass, 
    float dt, 
    int n)
{
    // Shared memory tile for this block
    __shared__ float shx[BLOCK_SIZE];
    __shared__ float shy[BLOCK_SIZE];
    __shared__ float shz[BLOCK_SIZE];
    __shared__ float shm[BLOCK_SIZE];

    // Precompute constants
    const float eps2 = SOFTENING * SOFTENING;
    const float Gdt = G * dt;
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load this particle's position into registers (fast access)
    float xi = 0.0f, yi = 0.0f, zi = 0.0f;
    if (i < n) {
        xi = px[i];
        yi = py[i];
        zi = pz[i];
    }
    
    // Force accumulators (in registers for speed)
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

    // Number of tiles needed to cover all particles
    int numTiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Iterate through all tiles
    for (int t = 0; t < numTiles; ++t) {
        // Coalesced load: each thread loads one particle into shared memory
        int loadIdx = t * BLOCK_SIZE + threadIdx.x;
        if (loadIdx < n) {
            shx[threadIdx.x] = px[loadIdx];
            shy[threadIdx.x] = py[loadIdx];
            shz[threadIdx.x] = pz[loadIdx];
            shm[threadIdx.x] = mass[loadIdx];
        } else {
            // Padding for out-of-bounds (zero mass = no force contribution)
            shx[threadIdx.x] = xi;
            shy[threadIdx.x] = yi;
            shz[threadIdx.x] = zi;
            shm[threadIdx.x] = 0.0f;
        }
        
        // Synchronize to ensure all data is loaded
        __syncthreads();

        // Compute forces from all particles in this tile
        if (i < n) {
            #pragma unroll UNROLL_FACTOR
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                // Distance vector
                float dx = shx[k] - xi;
                float dy = shy[k] - yi;
                float dz = shz[k] - zi;
                
                // Distance squared with softening
                float distSqr = dx*dx + dy*dy + dz*dz + eps2;
                
                // Fast inverse square root (hardware accelerated)
                float invDist = rsqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;
                
                // Force magnitude (m / r^3)
                float f = shm[k] * invDist3;
                
                // Accumulate force components
                Fx += f * dx;
                Fy += f * dy;
                Fz += f * dz;
            }
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }

    // Update velocities (F = ma, a = dv/dt)
    if (i < n) {
        vx[i] += Fx * Gdt;
        vy[i] += Fy * Gdt;
        vz[i] += Fz * Gdt;
    }
}

/**
 * Position integration kernel (Euler method)
 * Simple but efficient - just update positions based on velocities
 */
__global__ void integrateKernel(
    float* __restrict__ px, 
    float* __restrict__ py, 
    float* __restrict__ pz,
    const float* __restrict__ vx, 
    const float* __restrict__ vy, 
    const float* __restrict__ vz,
    float dt, 
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    px[i] += vx[i] * dt;
    py[i] += vy[i] * dt;
    pz[i] += vz[i] * dt;
}

// ═══════════════════════════════════════════════════════════════════════════════
//                              RENDERING
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Convert velocity to color (HSV-like mapping)
 * Slow particles: Blue
 * Medium particles: Green/Yellow  
 * Fast particles: Red/White
 */
void velocityToColor(float vx, float vy, float mass, Uint8* r, Uint8* g, Uint8* b) {
    float speed = sqrtf(vx*vx + vy*vy);
    
    // Normalize speed (adjust these based on your simulation)
    float t = fminf(speed / 3.0f, 1.0f);
    
    // Check if it's a massive body (black hole / central mass)
    if (mass > 100.0f) {
        // White/Yellow core
        *r = 255;
        *g = (Uint8)(255 - fminf(mass / 50.0f, 155.0f));
        *b = (Uint8)(100 + (rand() % 50));  // Slight flicker
        return;
    }
    
    // Color gradient: Blue -> Cyan -> Green -> Yellow -> Red -> White
    if (t < 0.25f) {
        // Blue to Cyan
        float s = t / 0.25f;
        *r = (Uint8)(50 * s);
        *g = (Uint8)(100 + 155 * s);
        *b = 255;
    } else if (t < 0.5f) {
        // Cyan to Green
        float s = (t - 0.25f) / 0.25f;
        *r = (Uint8)(50 + 50 * s);
        *g = 255;
        *b = (Uint8)(255 - 155 * s);
    } else if (t < 0.75f) {
        // Green to Yellow
        float s = (t - 0.5f) / 0.25f;
        *r = (Uint8)(100 + 155 * s);
        *g = 255;
        *b = (Uint8)(100 - 100 * s);
    } else {
        // Yellow to White
        float s = (t - 0.75f) / 0.25f;
        *r = 255;
        *g = 255;
        *b = (Uint8)(s * 200);
    }
}

/**
 * Render all particles with depth-based alpha and velocity colors
 */
void renderParticles(SDL_Renderer* renderer, Bodies* p, int n, Camera* cam) {
    float minZ = -50.0f, maxZ = 50.0f;
    
    for (int i = 0; i < n; ++i) {
        // Transform position with camera
        float sx = (p->x[i] + cam->offsetX) * cam->scale + WINDOW_WIDTH / 2.0f;
        float sy = (p->y[i] + cam->offsetY) * cam->scale + WINDOW_HEIGHT / 2.0f;
        
        // Culling - skip particles outside window
        if (sx < -10 || sx > WINDOW_WIDTH + 10 || sy < -10 || sy > WINDOW_HEIGHT + 10) {
            continue;
        }
        
        // Depth-based alpha (farther = dimmer)
        float depth = (p->z[i] - minZ) / (maxZ - minZ);
        depth = fmaxf(0.0f, fminf(1.0f, depth));
        float alpha = 0.3f + 0.7f * (1.0f - fabsf(depth - 0.5f) * 2.0f);
        
        // Get color based on velocity
        Uint8 r, g, b;
        velocityToColor(p->vx[i], p->vy[i], p->mass[i], &r, &g, &b);
        
        // Apply alpha
        r = (Uint8)(r * alpha);
        g = (Uint8)(g * alpha);
        b = (Uint8)(b * alpha);
        
        SDL_SetRenderDrawColor(renderer, r, g, b, 255);
        
        // Draw point (or small rect for massive bodies)
        if (p->mass[i] > 500.0f) {
            // Draw larger for massive bodies
            SDL_FRect rect = {sx - 2, sy - 2, 5, 5};
            SDL_RenderFillRect(renderer, &rect);
        } else if (p->mass[i] > 50.0f) {
            SDL_FRect rect = {sx - 1, sy - 1, 3, 3};
            SDL_RenderFillRect(renderer, &rect);
        } else {
            SDL_RenderPoint(renderer, sx, sy);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//                                  MAIN
// ═══════════════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[]) {
    // Print configuration
    printf("\n");
    printf("  ╔═══════════════════════════════════════════════════════════════╗\n");
    printf("  ║         CUDA N-BODY SIMULATION (BRUTE FORCE O(N²))           ║\n");
    printf("  ╠═══════════════════════════════════════════════════════════════╣\n");
    printf("  ║  Particles: %-6d    Block Size: %-3d                        ║\n", N, BLOCK_SIZE);
    printf("  ╠═══════════════════════════════════════════════════════════════╣\n");
    printf("  ║  CONTROLS:                                                    ║\n");
    printf("  ║    1 - Spiral Galaxy       3 - Random Chaos                  ║\n");
    printf("  ║    2 - Galaxy Collision    4 - Ring System (Saturn)          ║\n");
    printf("  ║    R - Reset    SPACE - Pause    +/- - Zoom    Arrows - Pan  ║\n");
    printf("  ║    ESC - Exit & Show Report                                  ║\n");
    printf("  ╚═══════════════════════════════════════════════════════════════╝\n");

    // Query GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0), "Get device properties");
    printf("GPU: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Memory: %.1f GB, SMs: %d\n\n", 
           prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f), 
           prop.multiProcessorCount);

    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL Init failed: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow(
        "CUDA N-Body Simulation", 
        WINDOW_WIDTH, WINDOW_HEIGHT, 
        SDL_WINDOW_RESIZABLE);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, NULL);
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);

    // Initialize data
    Bodies h_bodies, d_bodies;
    allocateBodiesHost(&h_bodies, N);
    allocateBodiesDevice(&d_bodies, N);

    srand((unsigned int)time(NULL));
    initBodies(&h_bodies, N, currentMode);
    copyBodiesToDevice(&d_bodies, &h_bodies, N);

    // Camera
    Camera cam = {0.0f, 0.0f, INITIAL_SCALE};

    // Kernel configuration
    const int GRID = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // State
    bool quit = false;
    bool paused = false;
    SDL_Event event;

    // Timing
    Uint64 lastTime = SDL_GetTicks();
    int frames = 0;
    long totalSteps = 0;
    char titleBuffer[256];

    // Performance tracking
    Uint64 startWallTime = SDL_GetTicks();
    double totalComputeMs = 0.0;
    double totalTransferMs = 0.0;

    cudaEvent_t startEvt, stopEvt, startXfer, stopXfer;
    cudaEventCreate(&startEvt);
    cudaEventCreate(&stopEvt);
    cudaEventCreate(&startXfer);
    cudaEventCreate(&stopXfer);

    printf("Simulation started. Press ESC to exit.\n");

    // Main loop
    while (!quit) {
        // Event handling
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) {
                quit = true;
            }
            if (event.type == SDL_EVENT_KEY_DOWN) {
                // Debug disabled - uncomment to see keycodes
                // printf("Key pressed: %d (0x%X)\\n", event.key.key, event.key.key);
                
                switch (event.key.key) {
                    case SDLK_ESCAPE:
                        quit = true;
                        break;
                    case SDLK_SPACE:
                        paused = !paused;
                        printf(paused ? "PAUSED\n" : "RESUMED\n");
                        break;
                    case SDLK_R:  // R key
                        // Reset simulation with current mode
                        srand((unsigned int)time(NULL));
                        initBodies(&h_bodies, N, currentMode);
                        copyBodiesToDevice(&d_bodies, &h_bodies, N);
                        cam.offsetX = 0; cam.offsetY = 0; cam.scale = INITIAL_SCALE;
                        totalSteps = 0;
                        totalComputeMs = 0;
                        totalTransferMs = 0;
                        startWallTime = SDL_GetTicks();
                        break;
                    // Number keys 1-4 to change mode
                    case SDLK_1:
                        currentMode = 1;
                        srand((unsigned int)time(NULL));
                        initBodies(&h_bodies, N, currentMode);
                        copyBodiesToDevice(&d_bodies, &h_bodies, N);
                        cam.offsetX = 0; cam.offsetY = 0; cam.scale = INITIAL_SCALE;
                        totalSteps = 0; totalComputeMs = 0; totalTransferMs = 0;
                        startWallTime = SDL_GetTicks();
                        break;
                    case SDLK_2:
                        currentMode = 2;
                        srand((unsigned int)time(NULL));
                        initBodies(&h_bodies, N, currentMode);
                        copyBodiesToDevice(&d_bodies, &h_bodies, N);
                        cam.offsetX = 0; cam.offsetY = 0; cam.scale = INITIAL_SCALE;
                        totalSteps = 0; totalComputeMs = 0; totalTransferMs = 0;
                        startWallTime = SDL_GetTicks();
                        break;
                    case SDLK_3:
                        currentMode = 3;
                        srand((unsigned int)time(NULL));
                        initBodies(&h_bodies, N, currentMode);
                        copyBodiesToDevice(&d_bodies, &h_bodies, N);
                        cam.offsetX = 0; cam.offsetY = 0; cam.scale = INITIAL_SCALE;
                        totalSteps = 0; totalComputeMs = 0; totalTransferMs = 0;
                        startWallTime = SDL_GetTicks();
                        break;
                    case SDLK_4:
                        currentMode = 4;
                        srand((unsigned int)time(NULL));
                        initBodies(&h_bodies, N, currentMode);
                        copyBodiesToDevice(&d_bodies, &h_bodies, N);
                        cam.offsetX = 0; cam.offsetY = 0; cam.scale = INITIAL_SCALE;
                        totalSteps = 0; totalComputeMs = 0; totalTransferMs = 0;
                        startWallTime = SDL_GetTicks();
                        break;
                    case SDLK_EQUALS:  // +
                    case SDLK_KP_PLUS:
                        cam.scale *= 1.2f;
                        break;
                    case SDLK_MINUS:
                    case SDLK_KP_MINUS:
                        cam.scale /= 1.2f;
                        break;
                    case SDLK_UP:
                        cam.offsetY += 10.0f / cam.scale;
                        break;
                    case SDLK_DOWN:
                        cam.offsetY -= 10.0f / cam.scale;
                        break;
                    case SDLK_LEFT:
                        cam.offsetX += 10.0f / cam.scale;
                        break;
                    case SDLK_RIGHT:
                        cam.offsetX -= 10.0f / cam.scale;
                        break;
                }
            }
        }

        float physTimeMs = 0;
        float xferTimeMs = 0;

        // Physics step (if not paused)
        if (!paused) {
            // Time GPU computation
            cudaEventRecord(startEvt);

            bodyForceKernel<<<GRID, BLOCK_SIZE>>>(
                d_bodies.x, d_bodies.y, d_bodies.z,
                d_bodies.vx, d_bodies.vy, d_bodies.vz,
                d_bodies.mass, DT, N);

            integrateKernel<<<GRID, BLOCK_SIZE>>>(
                d_bodies.x, d_bodies.y, d_bodies.z,
                d_bodies.vx, d_bodies.vy, d_bodies.vz,
                DT, N);

            cudaEventRecord(stopEvt);
            cudaEventSynchronize(stopEvt);
            cudaEventElapsedTime(&physTimeMs, startEvt, stopEvt);
            totalComputeMs += physTimeMs;

            // Time data transfer
            cudaEventRecord(startXfer);
            copyPositionsToHost(&h_bodies, &d_bodies, N);
            copyVelocitiesToHost(&h_bodies, &d_bodies, N);
            cudaEventRecord(stopXfer);
            cudaEventSynchronize(stopXfer);
            cudaEventElapsedTime(&xferTimeMs, startXfer, stopXfer);
            totalTransferMs += xferTimeMs;

            totalSteps++;
        }

        // Render - background color varies by mode
        int bgR = 5, bgG = 5, bgB = 15;
        switch(currentMode) {
            case 1: bgR = 5; bgG = 5; bgB = 20; break;    // Deep blue for spiral
            case 2: bgR = 15; bgG = 5; bgB = 10; break;   // Dark purple for collision  
            case 3: bgR = 10; bgG = 10; bgB = 10; break;  // Dark gray for chaos
            case 4: bgR = 5; bgG = 10; bgB = 15; break;   // Teal for rings
        }
        SDL_SetRenderDrawColor(renderer, bgR, bgG, bgB, 255);
        SDL_RenderClear(renderer);

        renderParticles(renderer, &h_bodies, N, &cam);
        
        SDL_RenderPresent(renderer);

        // Update window title with stats
        frames++;
        Uint64 currentTime = SDL_GetTicks();
        if (currentTime - lastTime >= 500) {
            float fps = (float)frames * 1000.0f / (float)(currentTime - lastTime);
            snprintf(titleBuffer, sizeof(titleBuffer),
                "[%d] %s | N=%d | Step: %ld | GPU: %.1f ms | FPS: %.0f %s",
                currentMode, getModeNameByIndex(currentMode),
                N, totalSteps, physTimeMs, fps, paused ? "[PAUSED]" : "");
            SDL_SetWindowTitle(window, titleBuffer);
            frames = 0;
            lastTime = currentTime;
        }
    }

    // Performance report
    Uint64 endWallTime = SDL_GetTicks();
    double totalWallSecs = (double)(endWallTime - startWallTime) / 1000.0;

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("                     PERFORMANCE REPORT\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("Total simulation steps:      %ld\n", totalSteps);
    printf("Total wall time:             %.2f seconds\n", totalWallSecs);
    printf("Total GPU compute time:      %.2f seconds\n", totalComputeMs / 1000.0);
    printf("Total transfer time:         %.2f seconds\n", totalTransferMs / 1000.0);
    printf("───────────────────────────────────────────────────────────────\n");
    
    if (totalSteps > 0) {
        double avgCompute = totalComputeMs / totalSteps;
        double avgTransfer = totalTransferMs / totalSteps;
        double interactionsPerSec = ((double)N * N * totalSteps) / (totalComputeMs / 1000.0);
        
        printf("Average GPU time/step:       %.3f ms\n", avgCompute);
        printf("Average transfer time/step:  %.3f ms\n", avgTransfer);
        printf("Interactions per second:     %.2e\n", interactionsPerSec);
        printf("Effective GFLOPS:            %.2f\n", 
               (interactionsPerSec * 20.0) / 1e9);  // ~20 FLOPs per interaction
    }
    
    printf("═══════════════════════════════════════════════════════════════\n");

    // Cleanup
    cudaEventDestroy(startEvt);
    cudaEventDestroy(stopEvt);
    cudaEventDestroy(startXfer);
    cudaEventDestroy(stopXfer);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    freeBodiesDevice(&d_bodies);
    freeBodiesHost(&h_bodies);

    return 0;
}
