#include "core/graph.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include <unordered_map>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        auto rebuildConnections = [this]()
        {
            for (const auto &tensor : tensors)
            {
                tensor->targets.clear();
                tensor->source.reset();
            }
            for (const auto &op : ops)
            {
                op->predecessors.clear();
                op->successors.clear();
            }
            for (const auto &op : ops)
            {
                for (const auto &input : op->inputs)
                    input->addTarget(op);
                for (const auto &output : op->outputs)
                    output->setSource(op);
            }
            for (const auto &op : ops)
                for (const auto &input : op->inputs)
                    if (auto pred = input->getSource())
                    {
                        pred->addSuccessors(op);
                        op->addPredecessors(pred);
                    }
        };

        // Remove pairs of transposes whose permutations compose to identity.
        bool changed = true;
        while (changed)
        {
            changed = false;
            for (const auto &firstOp : ops)
            {
                auto first = as<TransposeObj>(firstOp);
                if (!first)
                    continue;
                auto middle = first->getOutput();
                auto targets = middle->getTargets();
                if (targets.size() != 1)
                    continue;
                auto second = as<TransposeObj>(targets[0]);
                if (!second)
                    continue;

                const auto p = first->getPermute();
                const auto q = second->getPermute();
                bool inverse = p.size() == q.size();
                for (size_t i = 0; inverse && i < p.size(); ++i)
                    inverse = q[i] >= 0 && q[i] < static_cast<int>(p.size()) &&
                              p[q[i]] == static_cast<int>(i);
                if (!inverse)
                    continue;

                auto original = first->getInputs(0);
                auto redundant = second->getOutput();
                auto consumers = redundant->getTargets();
                if (consumers.empty())
                    continue;
                for (const auto &consumer : consumers)
                    consumer->replaceInput(redundant, original);

                ops.erase(std::remove(ops.begin(), ops.end(), firstOp), ops.end());
                ops.erase(std::remove(ops.begin(), ops.end(), targets[0]), ops.end());
                tensors.erase(std::remove(tensors.begin(), tensors.end(), middle),
                              tensors.end());
                tensors.erase(std::remove(tensors.begin(), tensors.end(), redundant),
                              tensors.end());
                rebuildConnections();
                changed = true;
                break;
            }
        }

        // Fold a last-two-axis transpose into the corresponding Matmul flag.
        changed = true;
        while (changed)
        {
            changed = false;
            for (const auto &op : ops)
            {
                auto matmul = as<MatmulObj>(op);
                if (!matmul)
                    continue;
                for (size_t inputIndex = 0; inputIndex < 2; ++inputIndex)
                {
                    auto transposed = matmul->getInputs(inputIndex);
                    auto transpose = as<TransposeObj>(transposed->getSource());
                    if (!transpose || transposed->getTargets().size() != 1)
                        continue;
                    const auto perm = transpose->getPermute();
                    bool swapsLastTwo = perm.size() >= 2;
                    for (size_t i = 0; swapsLastTwo && i + 2 < perm.size(); ++i)
                        swapsLastTwo = perm[i] == static_cast<int>(i);
                    if (swapsLastTwo)
                    {
                        const auto rank = perm.size();
                        swapsLastTwo = perm[rank - 2] == static_cast<int>(rank - 1) &&
                                       perm[rank - 1] == static_cast<int>(rank - 2);
                    }
                    if (!swapsLastTwo)
                        continue;

                    matmul->replaceInput(transposed, transpose->getInputs(0));
                    if (inputIndex == 0)
                        matmul->setTransA(!matmul->getTransA());
                    else
                        matmul->setTransB(!matmul->getTransB());
                    auto inferred = matmul->inferShape(matmul->getInputs());
                    IT_ASSERT(inferred.has_value() && inferred->size() == 1 &&
                              inferred->at(0) == matmul->getOutput()->getDims());

                    auto transposeOp = transposed->getSource();
                    ops.erase(std::remove(ops.begin(), ops.end(), transposeOp),
                              ops.end());
                    tensors.erase(std::remove(tensors.begin(), tensors.end(), transposed),
                                  tensors.end());
                    rebuildConnections();
                    changed = true;
                    break;
                }
                if (changed)
                    break;
            }
        }

        sorted = false;
        IT_ASSERT(topo_sort());
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        std::unordered_map<TensorObj *, size_t> offsets;
        std::unordered_map<TensorObj *, size_t> remainingUses;
        for (const auto &tensor : tensors)
            remainingUses[tensor.get()] = tensor->getTargets().size();

        // Inputs are populated before execution, so they must all be live from
        // the start even if their first consumers occur later in the graph.
        for (const auto &input : getInputs())
            offsets[input.get()] = allocator.alloc(input->getBytes());

        for (const auto &op : ops)
        {
            for (const auto &output : op->getOutputs())
                if (offsets.find(output.get()) == offsets.end())
                    offsets[output.get()] = allocator.alloc(output->getBytes());

            for (const auto &input : op->getInputs())
            {
                auto &uses = remainingUses[input.get()];
                IT_ASSERT(uses > 0);
                --uses;
                if (uses == 0)
                    allocator.free(offsets.at(input.get()), input->getBytes());
            }
        }

        auto *base = static_cast<std::byte *>(allocator.getPtr());
        for (const auto &tensor : tensors)
            tensor->setDataBlob(
                make_ref<BlobObj>(runtime, base + offsets.at(tensor.get())));

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini
