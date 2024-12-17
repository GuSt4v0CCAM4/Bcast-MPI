#include <iomanip>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include "binomial_bcast.h"
#include "linear_bcast.h"
#include "binary_bcast.h"
#include "binomial_bcast_one_sided.h"
#include "binary_bcast_one_side.h"

#define number_of_messages 20
#define start_length 4
#define length_factor 2
#define max_length 128000
#define number_package_sizes 14

enum bcast_types_t
{
    linear = 1,
    binomial = 2,
    binary = 3,
    binomialOne = 4,
    binaryOne = 5
};
bcast_types_t bcast_type = linear;

void initialize_send_buffer(buf_dtype *snd_buf, int test_value, int length, int message_number)
{
    int mid = (length - 1) / number_of_messages * message_number;
    if (mid >= length) mid = length - 1;

    snd_buf[0] = test_value + 1;
    snd_buf[mid] = test_value + 2;
    snd_buf[length - 1] = test_value + 3;
}

int MPI_initialization(int *argc, char ***argv, MPI_Comm *comm_sm)
{
    int provided = MPI_THREAD_MULTIPLE;
    setbuf(stdout, NULL);
    MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);

    return MPI_Comm_split(MPI_COMM_WORLD, 0, 0, comm_sm);
}

int main(int argc, char *argv[])
{
    MPI_Comm comm_sm;
    int result = MPI_initialization(&argc, &argv, &comm_sm);
    if (result != MPI_SUCCESS)
        MPI_Abort(MPI_COMM_WORLD, result);

    if (argc < 2)
    {
        std::cerr << "Error: Argumento de difusión no proporcionado." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (std::string(argv[1]) == "linear")
        bcast_type = linear;
    else if (std::string(argv[1]) == "binomial")
        bcast_type = binomial;
    else if (std::string(argv[1]) == "binary")
        bcast_type = binary;
    else if (std::string(argv[1]) == "binomialOne")
        bcast_type = binomialOne;
    else if (std::string(argv[1]) == "binaryOne")
        bcast_type = binaryOne;
    else
    {
        std::cerr << "Error: Tipo de difusión desconocido." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::string algname = argv[1];
    int my_rank, size;
    MPI_Comm_rank(comm_sm, &my_rank);
    MPI_Comm_size(comm_sm, &size);

    // Crear ventana de memoria compartida
    MPI_Win win;
    buf_dtype *shared_buf;
    result = MPI_Win_allocate((MPI_Aint)max_length * sizeof(buf_dtype), sizeof(buf_dtype),
                              MPI_INFO_NULL, comm_sm, &shared_buf, &win);
    if (result != MPI_SUCCESS)
        MPI_Abort(comm_sm, result);

    if (my_rank == 0)
    {
        printf("    message size      transfertime  duplex bandwidth per process and neighbor\n");
    }

    // Variables para prueba
    double start, finish, transfer_time;
    int length = start_length, test_value;
    descr_t descr;
    descr.root = 0;
    buf_dtype snd_buf[max_length];

    for (int j = 1; j <= number_package_sizes; j++)
    {
        for (int i = 0; i < number_of_messages; i++)
        {
            if (i == 1) start = MPI_Wtime();
            test_value = j * 1000000 + i * 10000 + my_rank * 10;
            initialize_send_buffer(snd_buf, test_value, length, i);

            // Difusión optimizada
            if (my_rank == 0)
            {
                for (int target_rank = 1; target_rank < size; target_rank++)
                {
                    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target_rank, 0, win);
                    MPI_Put(snd_buf, length, MPI_FLOAT, target_rank, 0, length, MPI_FLOAT, win);
                    MPI_Win_unlock(target_rank, win);
                }
            }
            else
            {
                MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
                MPI_Get(shared_buf, length, MPI_FLOAT, 0, 0, length, MPI_FLOAT, win);
                MPI_Win_unlock(0, win);
            }

            // Sincronización local de la ventana
            MPI_Win_fence(0, win);
        }
        finish = MPI_Wtime();
        if (my_rank == 0)
        {
            transfer_time = (finish - start) / number_of_messages;
            printf("%10i bytes %12.3f usec %13.3f MB/s\n",
                   length * (int)sizeof(float), transfer_time * 1e6,
                   1.0e-6 * 2 * length * sizeof(float) / transfer_time);
        }
        length *= length_factor;
    }

    // Liberar ventana y finalizar
    MPI_Win_free(&win);
    MPI_Finalize();
}
