#include <iostream>
#include <mpi.h>

int main(int argc, char **argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (!rank) {
        for (int i = 15; i >= 1; --i) {
            int *buf = (int *) malloc(i * sizeof(*buf));
            for (int j = 0; j < i; ++j) {
                buf[j] = j + 1;
            }

            int dst = i % 4 == 0 ? 4 : 1;
            // MPI_Send(&i, 1, MPI_INT, dst, 1, MPI_COMM_WORLD);
            MPI_Send(buf, i, MPI_INT, dst, 1, MPI_COMM_WORLD);
        }

        std::cout << "Process 0 sent all messages and finished!" << std::endl;
        MPI_Finalize();
    } else {
        int dst = 0;
        int src = rank > 3 ? rank - 4 : rank - 1;

        MPI_Status status;
        while (1) {
            // MPI_Recv(&dst, 1, MPI_INT, src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Probe(src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int message_size;
            MPI_Get_count(&status, MPI_INT, &message_size);

            int *buf = (int *) malloc(message_size * sizeof(*buf));
            MPI_Recv(buf, dst, MPI_INT, src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (dst == rank) {
                std::cout << "Process "<< rank << " message: ";
                for (int i = 0; i < dst; ++i) {
                    std::cout << buf[i] << " ";
                }
                std::cout << std::endl;

                MPI_Finalize();
                break;
            } else {
                int neighbor = (dst - rank) % 4 == 0 ? rank + 4 : rank + 1;
                // MPI_Send(&dst, 1, MPI_INT, neighbor, 1, MPI_COMM_WORLD);
                MPI_Send(buf, dst, MPI_INT, neighbor, 1, MPI_COMM_WORLD);
            }
            free(buf);
        }
    }

    return 0;
}