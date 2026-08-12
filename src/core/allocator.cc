#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        IT_ASSERT(size > 0);
        auto best = freeBlocks.end();
        for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it)
            if (it->second >= size &&
                (best == freeBlocks.end() || it->second < best->second))
                best = it;

        if (best != freeBlocks.end())
        {
            const auto addr = best->first;
            const auto blockSize = best->second;
            freeBlocks.erase(best);
            if (blockSize > size)
                freeBlocks.emplace(addr + size, blockSize - size);
            return addr;
        }

        const auto addr = used;
        used += size;
        peak = std::max(peak, used);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================

        return addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        IT_ASSERT(size > 0);
        IT_ASSERT(addr % alignment == 0 && addr + size <= used);

        auto next = freeBlocks.lower_bound(addr);
        if (next != freeBlocks.end() && addr + size == next->first)
        {
            size += next->second;
            freeBlocks.erase(next);
        }

        auto right = freeBlocks.lower_bound(addr);
        if (right != freeBlocks.begin())
        {
            auto prev = std::prev(right);
            IT_ASSERT(prev->first + prev->second <= addr);
            if (prev->first + prev->second == addr)
            {
                addr = prev->first;
                size += prev->second;
                freeBlocks.erase(prev);
            }
        }

        if (addr + size == used)
            used = addr;
        else
            freeBlocks.emplace(addr, size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        if (size == 0)
            return 0;
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
