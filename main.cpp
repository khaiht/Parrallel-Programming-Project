#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <SDL3/SDL.h>

// --- SỐ THREAD: TỰ ĐỘNG DETECT MAX --- 

// --- THAM SỐ MÔ PHỎNG ---
#define N 10000           // Số lượng hạt (ĐỒNG BỘ VỚI CUDA)
#define MAX_STEPS 1000    // Step
#define G 1.0f            // Hằng số hấp dẫn (ĐỒNG BỘ VỚI CUDA)
#define SOFTENING 0.1f    // Tránh chia cho 0 khi hai hạt quá gần nhau (ĐỒNG BỘ VỚI CUDA)
#define DT 0.005f         // Bước thời gian

// --- THAM SỐ ĐỒ HỌA ---
#define WINDOW_WIDTH 1200
#define WINDOW_HEIGHT 900
#define SCALE 150.0f      

// No boundary - particles move freely (đồng bộ với CUDA)

typedef struct {
    float x, y, z;
    float vx, vy, vz;
    float mass;
} Body;

void initBodies(Body* p) {
    // SPIRAL GALAXY - đồng bộ với CUDA mode 1
    for (int i = 0; i < N; i++) {
        float angle = ((float)rand() / RAND_MAX) * 2.0f * 3.14159265f;
        float radius = ((float)rand() / RAND_MAX) * 2.0f;
        
        p[i].x = radius * cosf(angle);
        p[i].y = radius * sinf(angle);
        p[i].z = ((float)rand() / RAND_MAX * 0.2f - 0.1f);

        float speed = sqrtf(radius) * 0.3f;
        p[i].vx = -sinf(angle) * speed; 
        p[i].vy = cosf(angle) * speed;
        p[i].vz = 0.0f;

        p[i].mass = 1.0f;
    }
}

void computeForces(Body* bodies) {
    const float eps2 = SOFTENING * SOFTENING;

    #pragma omp parallel for schedule(static) //Chia vòng lặp cho các luồng xử lý song song
    for (int i = 0; i < N; i++) {
        float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

        for (int j = 0; j < N; j++) {
            if (i == j) continue;

            float dx = bodies[j].x - bodies[i].x;
            float dy = bodies[j].y - bodies[i].y;
            float dz = bodies[j].z - bodies[i].z;

            float distSqr = dx*dx + dy*dy + dz*dz + eps2;
            float invDist = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            float f = G * bodies[i].mass * bodies[j].mass * invDist3;

            Fx += f * dx; Fy += f * dy; Fz += f * dz;
        }

        bodies[i].vx += (Fx / bodies[i].mass) * DT;
        bodies[i].vy += (Fy / bodies[i].mass) * DT;
        bodies[i].vz += (Fz / bodies[i].mass) * DT;
    }
}

void integratePositions(Body* bodies) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        // Cập nhật vị trí (NO BOUNDARY - đồng bộ với CUDA)
        bodies[i].x += bodies[i].vx * DT;
        bodies[i].y += bodies[i].vy * DT;
        bodies[i].z += bodies[i].vz * DT;
    }
}

int main(int argc, char* argv[]) {
    int max_procs = omp_get_num_procs();
    int num_threads = max_procs;  // Tự động dùng tất cả cores
    omp_set_num_threads(num_threads);

    printf("=== N-BODY SIMULATION - OpenMP (CPU) ===\n");
    printf("Particles: %d\n", N);
    printf("CPU Cores Available: %d\n", max_procs);
    printf("Threads Using: %d (MAX)\n", num_threads);
    printf("G=%.1f, Softening=%.2f, DT=%.4f\n", G, SOFTENING, DT);
    printf("=========================================\n\n");

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL Error: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("N-Body OpenMP (CPU)",
                                          WINDOW_WIDTH, WINDOW_HEIGHT, 0);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, NULL);

    Body* bodies = (Body*)malloc(N * sizeof(Body));
    if (bodies == NULL) return 1;
    
    srand((unsigned)time(NULL));
    initBodies(bodies);

    int running = 1;
    SDL_Event event;
    
    double totalStart = omp_get_wtime(); 
    long frameCount = 0;
    char titleBuffer[256];

    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) running = 0;
            if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE) running = 0;
        }

        double startPhys = omp_get_wtime();

        computeForces(bodies);
        integratePositions(bodies); // Đã có xử lý va chạm bên trong hàm này

        double endPhys = omp_get_wtime();
        double physTimeMS = (endPhys - startPhys) * 1000.0;

        // Render
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        // Render hạt
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        for (int i = 0; i < N; i++) {
            int sx = (int)(bodies[i].x * SCALE + WINDOW_WIDTH / 2);
            int sy = (int)(bodies[i].y * SCALE + WINDOW_HEIGHT / 2);
            
            // Vẽ hạt (chỉ vẽ nếu còn nằm trong vùng hiển thị)
            if (sx >= 0 && sx < WINDOW_WIDTH && sy >= 0 && sy < WINDOW_HEIGHT) {
                SDL_RenderPoint(renderer, (float)sx, (float)sy);
            }
        }
        SDL_RenderPresent(renderer);

        frameCount++;

        if (frameCount % 60 == 0) {
            double fps = 1000.0 / (physTimeMS + 16.0);
            // Hiển thị thông số trên chương trình
            sprintf(titleBuffer, "Step: %ld | Threads: %d | Phys: %.3f ms | FPS: %.1f", 
                    frameCount, omp_get_max_threads(), physTimeMS, fps);
            SDL_SetWindowTitle(window, titleBuffer);
        }

        if (frameCount >= MAX_STEPS) {
            printf("\n--- DA HOAN THANH %d BUOC MO PHONG ---\n", MAX_STEPS);
            running = 0; 
        }

    }

    double totalEnd = omp_get_wtime();
    printf("Tong thoi gian chay: %.2f giay.\n", totalEnd - totalStart);

    free(bodies);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}