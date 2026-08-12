#include "operators/transpose.h"

namespace infini
{
    TransposeObj::TransposeObj(GraphObj *graph, Tensor input, Tensor output,
                               vector<int> permute)
        : OperatorObj(OpType::Transpose, {input}, {output})
    {
        auto rank = input->getRank();
        if (permute.empty())
        {
            transposePermute.resize(rank);
            for (size_t i = 0; i < rank; ++i)
            {
                transposePermute[i] = static_cast<int>(rank - 1 - i);
            }
        }
        else
        {
            IT_ASSERT(rank == permute.size());
            transposePermute = std::move(permute);
        }
        IT_ASSERT(checkValid(graph));
    }

    optional<vector<Shape>> TransposeObj::inferShape(const TensorVec &inputs)
    {
        const auto A = inputs[0];
        auto input_dim = A->getDims();
        int rank = A->getRank();
        Shape output_dim(rank);

        IT_ASSERT(static_cast<int>(transposePermute.size()) == rank);
        vector<bool> seen(rank, false);
        for (int i = 0; i < rank; ++i)
        {
            const int axis = transposePermute[i];
            IT_ASSERT(axis >= 0 && axis < rank && !seen[axis]);
            seen[axis] = true;
            output_dim[i] = input_dim[axis];
        }

        // =================================== 作业 ===================================
        // TODO：修改 output_dim，返回正确的 transpose 后的 shape
        // REF: https://onnx.ai/onnx/operators/onnx__Transpose.html#transpose-21
        // =================================== 作业 ===================================

        return {{output_dim}};
    }

    std::string TransposeObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << vecToString(inputs[0]->getDims()) << ",";
        os << "input=" << inputs[0]->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }
}; // namespace infini
