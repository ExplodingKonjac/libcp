#include <map>

#ifdef DADALZY
int test = 1;  // 如果这段代码在 VSCode 里是亮起的（未被灰显），说明 .clangd
               // 成功加载了。
#endif

int main() {
    std::map<int, int> mp();
    return 0;
}