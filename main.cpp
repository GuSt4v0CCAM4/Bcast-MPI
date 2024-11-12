#include <iomanip>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include "binomial_bcast.h"
#include "linear_bcast.h"
#include "binary_bcast.h"
#include "binomial_bcast_one_sided.h"
#include "binary_bcast_one_side.h"

// #include "gtest/gtest.h"
// #include "tests.cpp"
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
    test = 4,
    binomialOne = 5,
    binaryOne = 6
};
bcast_types_t bcast_type = linear;

void intialize_send_buffer(buf_dtype *snd_buf, int test_value, int length, int message_number)
{
    // Asegurarse de que mid esté dentro de los límites
    int mid = (length - 1) / number_of_messages * message_number;
    if (mid >= length) mid = length - 1;  // Evitar acceso fuera de los límites del buffer

    snd_buf[0] = test_value + 1;
    snd_buf[mid] = test_value + 2;
    snd_buf[length - 1] = test_value + 3;
}

int MPI_intialization(int *argc, char ***argv, MPI_Comm *comm_sm)
{
    int provided = MPI_THREAD_MULTIPLE;
    setbuf(stdout, NULL);
    MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);

    return MPI_Comm_split(MPI_COMM_WORLD, 0, 0, comm_sm);
}

int main(int argc, char *argv[])
{
    // Inicializar MPI antes de verificar los argumentos
    MPI_Comm comm_sm;
    int result = MPI_intialization(&argc, &argv, &comm_sm);
    if (result != MPI_SUCCESS)
        MPI_Abort(MPI_COMM_WORLD, result);

    // Verificar si el argumento de difusión es proporcionado
    if (argc < 2)
    {
        std::cerr << "Error: Argumento de difusión no proporcionado." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Seleccionar el tipo de difusión basado en el argumento
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
        std::cerr << "Error: Tipo de difusión desconocido. Usa 'linear', 'binomial', 'binary', 'binomialOne' o 'binaryOne'." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::string algname = argv[1];
    int my_rank;
    MPI_Comm_rank(comm_sm, &my_rank);

    int size;
    MPI_Comm_size(comm_sm, &size);

    // Asignar memoria para el buffer
    MPI_Aint buf_size;
    MPI_Win win;
    int disp_unit;
    buf_dtype *rcv_buf; // buffer de recepción
    result = MPI_Win_allocate((MPI_Aint)max_length * sizeof(buf_dtype), sizeof(buf_dtype), MPI_INFO_NULL, comm_sm, &rcv_buf, &win);
    if (result != MPI_SUCCESS)
        MPI_Abort(comm_sm, result);

    //? Declaración de archivo para guardar resultados
    std::fstream file;
    std::fstream fileBench;
    file.open("results/result" + std::string(argv[1]) + std::to_string(size) + ".dat", std::ios::out); // Crear archivo
    fileBench.open("results/result" + std::string(argv[1]) + std::to_string(size) + "Bench.dat", std::ios::out);

    if (my_rank == 0 && bcast_type != test)
    {
        printf("    message size      transfertime  duplex bandwidth per process and neighbor\n");
        fileBench << "    message size      transfertime  duplex bandwidth per process and neighbor" << std::endl;
    }

    // Variables de prueba para el Broadcast
    if (my_rank == 0 && bcast_type != test)
    {
        double start, finish, transfer_time;
        int i, length, test_value;
        length = start_length;
        descr_t descr;
        descr.root = 0;
        buf_dtype snd_buf[max_length];

        for (int j = 1; j <= number_package_sizes; j++)
        {
            for (i = 0; i <= number_of_messages; i++)
            {
                if (i == 1)
                    start = MPI_Wtime(); // Iniciar el temporizador
                test_value = j * 1000000 + i * 10000 + my_rank * 10;
                intialize_send_buffer(snd_buf, test_value, length, i);
                descr.message_length = length;

                // Selecciona la operación de difusión según el tipo
                if (bcast_type == binomial)
                    RMA_Bcast_binomial((buf_dtype *)snd_buf, rcv_buf, my_rank, descr, size, win, comm_sm);
                else if (bcast_type == binomialOne)
                    RMA_Bcast_binomial_OneSide((buf_dtype *)snd_buf, MPI_FLOAT, buf_size, my_rank, descr, size, win, comm_sm);
                else if (bcast_type == linear)
                    RMA_Bcast_Linear((buf_dtype *)snd_buf, MPI_FLOAT, buf_size, descr, size, win, comm_sm);
                else if (bcast_type == binary)
                    BinaryTreeBcast((buf_dtype *)snd_buf, rcv_buf, my_rank, descr, size, win, comm_sm);
                else if (bcast_type == binaryOne)
                    RMA_Bcast_binary_OneSide((buf_dtype *)snd_buf, MPI_FLOAT, buf_size, my_rank, descr, size, win, comm_sm);
            }
            finish = MPI_Wtime();
            if (my_rank == 0)
            {
                transfer_time = (finish - start) / number_of_messages; // Calcular el tiempo de transferencia por mensaje
                fileBench << std::setw(10) << length * sizeof(float) << " bytes " << std::setw(12) << transfer_time * 1e6 << " usec " << std::setw(13) << 1.0e-6 * 2 * length * sizeof(float) / transfer_time << " MB/s" << std::endl;
                printf("%10i bytes %12.3f usec %13.3f MB/s\n",
                       length * (int)sizeof(float), transfer_time * 1e6, 1.0e-6 * 2 * length * sizeof(float) / transfer_time);
            }
            length = length * length_factor;
        }
        MPI_Win_flush(my_rank, win);
    }
    else
    {
        if (bcast_type != test)
            MPI_Win_flush(my_rank, win);
    }

    // Liberar la memoria de la ventana
    MPI_Win_free(&win);

    // Liberar la memoria de MPI y finalizar
    MPI_Finalize();
}
