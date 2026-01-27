#include <random>

#include "testlib.h"

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    rnd.setSeed(std::random_device{}());

    int n = 1 << 14;
    std::cout << n << "\n0 ";
    for (int i = 1; i < n; i++) std::cout << rnd.next(0, 100) << ' ';
    std::cout << '\n';
    return 0;
}
