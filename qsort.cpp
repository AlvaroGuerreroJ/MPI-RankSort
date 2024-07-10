#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

#include "util.hpp"

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " N\n";
        return 1;
    }

    uint32_t target_size = std::stoul(argv[1]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(11, 99);

    int* my_data = new int[target_size];

    for (uint32_t i = 0; i < target_size; i++)
    {
        int rn = distrib(gen);
        my_data[i] = rn;
    }

    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;

    auto t0 = high_resolution_clock::now();

    std::sort(my_data, my_data + target_size);

    auto t_end = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t_end - t0;

    if (check_sorted(my_data, target_size))
    {
        std::cout << "Array of size " << target_size << " sorted properly\n";
    }
    else
    {
        std::cout << "Array of size " << target_size << " failed sorting\n";
    }

    std::cout << "Took: " << ms_double.count() << "\n";
}
