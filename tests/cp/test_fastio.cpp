#include <cassert>
#include <cstdio>
#include <string>

#include "cp/fast_io.hpp"

void test_string(size_t size) {
    FILE* file = std::tmpfile();
    assert(file);

    std::string expected = "prefix:";
    expected.append(size, 'x');
    expected += ":suffix";
    {
        cp::FastOutput out(file);
        out.print("prefix:");
        out.print(std::string(size, 'x'));
        out.print(":suffix");
    }

    std::rewind(file);
    std::string actual(expected.size(), '\0');
    assert(std::fread(actual.data(), 1, actual.size(), file) == actual.size());
    assert(std::fgetc(file) == EOF);
    assert(actual == expected);
    std::fclose(file);
}

int main() {
    test_string(cp::OUT_BUF_SIZE - 1);
    test_string(cp::OUT_BUF_SIZE);
    test_string(cp::OUT_BUF_SIZE + 1);
    test_string(cp::OUT_BUF_SIZE * 2 + 17);
    return 0;
}
