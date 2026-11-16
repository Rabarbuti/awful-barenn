#include <cstdio>
#include <omp.h>
#include <sched.h>

// Mostra em qual CPU logica cada thread do OpenMP foi parar.
int main() {
    #pragma omp parallel
    {
        #pragma omp critical
        std::printf("thread %2d de %2d na cpu %2d\n",
                    omp_get_thread_num(), omp_get_num_threads(), sched_getcpu());
    }
    return 0;
}
