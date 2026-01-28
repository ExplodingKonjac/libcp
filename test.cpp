#include <iostream>
#include <string_view>

template <size_t N>
struct FixedString {
    const char* data;
    consteval FixedString(const char (&str)[N]): data(str) {}
    constexpr auto view() const { return std::string_view{data, N - 1}; }
};

template <FixedString S>
void func() {
    std::cout << S.view() << '\n';
}

int main() { func<"abc">(); }