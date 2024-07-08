#include <algorithm>
#include <cassert>
#include <cstdint>
#include <random>

#include <mpi.h>

bool check_sorted(int* arr, int size)
{
    for (int i = 0; i < size - 1; i++)
    {
        if (arr[i] > arr[i + 1])
        {
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int my_rank;
    int np;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    // The rows are sorted and then compared with each column. The correct
    // indexes for the rows elements are the final result.
    uint32_t n_rows = 0;
    uint32_t n_cols = 0;

    std::map<uint32_t, std::pair<uint32_t, uint32_t>> mappings = {
        { 4, {2, 2}},
        { 6, {2, 3}},
        { 8, {2, 4}},
        { 9, {3, 3}},
        {10, {2, 5}},
        {12, {3, 4}},
    };

    auto mapping_found = mappings.find(np);

    if (mapping_found == mappings.end())
    {
        if (my_rank == 0)
        {
            std::cerr << "No distribution specified for " << np << " processors.\n";
        }

        MPI_Finalize();
        return 0;
    }

    std::tie(n_rows, n_cols) = mapping_found->second;

    uint32_t target_size = 100000000;

    while (target_size % np != 0)
    {
        target_size += 1;
    }
    uint32_t const size_per_proc = target_size / np;

    // Each process must have enough space to hold a copy of all the elements in
    // his row and column.
    uint32_t n_row_elements = size_per_proc * n_cols;
    int* row_elements = new int[n_row_elements];

    uint32_t n_col_elements = size_per_proc * n_rows;
    int* col_elements = new int[n_col_elements];

    // Processors start counting row by row
    uint32_t my_row = my_rank / n_cols;
    uint32_t my_col = my_rank % n_cols;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(11, 99);

    int* my_data = new int[size_per_proc];

    uint32_t my_row_start = my_col * size_per_proc;
    uint32_t my_col_start = my_row * size_per_proc;

    for (uint32_t i = 0; i < size_per_proc; i++)
    {
        int rn = distrib(gen);
        my_data[i] = rn;
    }

    uint32_t my_row_end = (my_col + 1) * size_per_proc;
    // std::sort(row_elements + my_row_start, row_elements + my_row_end);

    std::sort(my_data, my_data + size_per_proc);

    // for (uint32_t i = 0; i < size_per_proc; i++)
    // {
    //     col_elements[my_col_start + i] = row_elements[my_row_start + i];
    // }

    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, my_row, my_col, &row_comm);

    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, my_col, my_row, &col_comm);

    MPI_Barrier(MPI_COMM_WORLD);

    {
        int d_rows;
        MPI_Comm_size(row_comm, &d_rows);
        int d_cols;
        MPI_Comm_size(col_comm, &d_cols);

        // printf("R: %d %d\n", n_cols, d_rows);
        // printf("C: %d %d\n", n_rows, d_cols);

        assert(d_rows == n_cols);
        assert(d_cols == n_rows);
    }

    MPI_Allgather(
        my_data, size_per_proc, MPI_INT, row_elements, size_per_proc, MPI_INT, row_comm);

    MPI_Allgather(
        my_data, size_per_proc, MPI_INT, col_elements, size_per_proc, MPI_INT, col_comm);

    delete[] my_data;

    std::sort(row_elements, row_elements + n_row_elements);
    std::sort(col_elements, col_elements + n_col_elements);

    uint32_t* indexes_for_row = new uint32_t[n_row_elements];
    uint32_t cur_col_index = 0;

    for (uint32_t i = 0; i < n_row_elements; i++)
    {
        while (cur_col_index < n_col_elements && row_elements[i] > col_elements[cur_col_index])
        {
            cur_col_index += 1;
        }
        indexes_for_row[i] = cur_col_index;
    }

    uint32_t* reduced_indexes_for_row = new uint32_t[n_row_elements];
    MPI_Reduce(
        indexes_for_row,
        reduced_indexes_for_row,
        n_row_elements,
        MPI_UINT32_T,
        MPI_SUM,
        0,
        row_comm);

    if (my_rank == 0)
    {
        printf("Row:\n");
        for (uint32_t i = 0; i < n_row_elements; i++)
        {
            std::cout << row_elements[i] << " ";
        }
        std::cout << "\n";

        printf("Row indexes:\n");
        for (uint32_t i = 0; i < n_row_elements; i++)
        {
            std::cout << indexes_for_row[i] << " ";
        }
        std::cout << "\n";

        printf("Reduced row indexes:\n");
        for (uint32_t i = 0; i < n_row_elements; i++)
        {
            std::cout << reduced_indexes_for_row[i] << " ";
        }
        std::cout << "\n";

        printf("Col:\n");
        for (uint32_t i = 0; i < n_col_elements; i++)
        {
            std::cout << col_elements[i] << " ";
        }
        std::cout << "\n";

        uint32_t total_elements = n_row_elements * n_rows;
        int* all_data = new int[total_elements];

        for (uint32_t i = 0; i < total_elements; i++)
        {
            all_data[i] = 0;
        }

        for (uint32_t i = 0; i < n_row_elements; i++)
        {
            all_data[reduced_indexes_for_row[i]] = row_elements[i];
        }

        int* received_row = new int[n_row_elements];
        uint32_t* received_row_indexes = new uint32_t[n_row_elements];

        for (int i = 1; i < n_rows; i++)
        {
            MPI_Recv(
                received_row,
                n_row_elements,
                MPI_INT,
                i,
                MPI_ANY_TAG,
                col_comm,
                MPI_STATUS_IGNORE);

            MPI_Recv(
                received_row_indexes,
                n_row_elements,
                MPI_UINT32_T,
                i,
                MPI_ANY_TAG,
                col_comm,
                MPI_STATUS_IGNORE);

            for (uint32_t i = 0; i < n_row_elements; i++)
            {
                all_data[received_row_indexes[i]] = received_row[i];
            }
        }

        int last_seen = 0;
        for (uint32_t i = 0; i < total_elements; i++)
        {
            if (all_data[i] == 0)
            {
                all_data[i] = last_seen;
            }
            else
            {
                last_seen = all_data[i];
            }
        }
        // MPI_Gather(
        //     row_elements,
        //     static_cast<int>(n_row_elements),
        //     MPI_INT,
        //     all_data,
        //     static_cast<int>(total_elements),
        //     MPI_INT,
        //     0,
        //     col_comm);

        // MPI_Gather(
        //     reduced_indexes_for_row,
        //     n_row_elements,
        //     MPI_UINT32_T,
        //     all_data_indexes,
        //     total_elements,
        //     MPI_UINT32_T,
        //     0,
        //     col_comm);

        for (uint32_t i = 0; i < total_elements; i++)
        {
            printf("%.3i\t%i\n", i, all_data[i]);
        }

        if (check_sorted(all_data, total_elements))
        {
            std::cout << "Array of size " << total_elements << " sorted properly\n";
        }
        else
        {
            std::cout << "Array of size " << total_elements << " failed sorting\n";
        }

        delete[] all_data;
    }
    else if (my_col == 0)
    {
        MPI_Send(row_elements, n_row_elements, MPI_INT, 0, 0, col_comm);
        MPI_Send(reduced_indexes_for_row, n_row_elements, MPI_UINT32_T, 0, 0, col_comm);
    }

    delete[] row_elements;
    delete[] col_elements;
    delete[] indexes_for_row;
    delete[] reduced_indexes_for_row;

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    MPI_Finalize();
}
