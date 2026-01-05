#define CP_FASTIO_USE_BUF
#include "full/FastIO.hpp"

cp::FastInput qin;
cp::FastOutput qout;

int main() {
    auto T = qin.scan<int>().value();

    if (T == 1) {
        while (true) {
            auto x = qin.scan<int>();
            if (!x) break;
            qout.println(x.value_or(0));
        }
    } else {
        while (true) {
            auto x = qin.scan<std::string>();
            if (!x) break;
            qout.println(x.value_or(""));
        }
    }
    return 0;
}
