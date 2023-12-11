// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "prefetch_pass.hpp"
#include <queue>
#include "nnfusion/common/common.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/constant.hpp"
#include "nnfusion/core/operators/op_define/fused.hpp"
#include "nnfusion/core/operators/op_define/noop.hpp"
#include "nnfusion/engine/cache/manager.hpp"
#include "nnfusion/engine/op.hpp"
#include "nnfusion/engine/pass/graph/kernel_selection.hpp"
#include "nnfusion/engine/profiler/profiler.hpp"
#include "nnfusion/core/kernels/cuda_gpu/cuda_emitter.hpp"

using namespace nnfusion;
using namespace nnfusion::pass::graph;
using namespace nnfusion::profiler;

DEFINE_bool(fprefetch,
            true,
            "Enable prefetching for tensors, only works for CUDA GPU backend.");

// function parameter class
class CudaKernelParameter
{
public:
    CudaKernelParameter(const std::string name, const nnfusion::Shape shape,const nnfusion::element::Type type, bool is_input, size_t index)
        : m_name(name)
        , m_shape(shape)
        , m_type(type)
        , m_is_input(is_input)
        , m_index(index)
    {
    }

    std::string get_name() { return m_name; }
    const nnfusion::element::Type get_type() { return m_type; }
    nnfusion::Shape get_shape() { return m_shape; }
    bool is_input() { return m_is_input; }
    std::string get_name_in_kernel()
    {
        if (m_is_input)
            return "input" + std::to_string(m_index);
        else
            return "output" + std::to_string(m_index);
    }
private:
    std::string m_name;
    nnfusion::Shape m_shape;
    const nnfusion::element::Type m_type;
    bool m_is_input;
    size_t m_index;
};

class CudaKernel
{
public:
    CudaKernel(const std::shared_ptr<nnfusion::graph::GNode> gnode, const json kernel_json, size_t prefetch_begin_index = 0)
        : m_gnode(gnode)
        , m_kernel_json(kernel_json)
        , m_prefetch_begin_index(prefetch_begin_index)
    {
        set_parameters(gnode);
        m_code = m_kernel_json["code"].get<std::string>();
        m_grid_size = m_kernel_json["grid_size"].get<std::vector<size_t>>();
        m_block_size = m_kernel_json["block_size"].get<std::vector<size_t>>();
        m_total_thread_num = m_grid_size[0] * m_grid_size[1] * m_grid_size[2] * m_block_size[0] * m_block_size[1] * m_block_size[2];
    }

    // Set Input and Output
    // Attention: get_output_size() are not equal to get_out_edges(). get_out_edges() may contain more edges on one tensor.
    void set_parameters(std::shared_ptr<nnfusion::graph::GNode> gnode)
    {
        kernel_inputs.clear();
        kernel_outputs.clear();

        for (size_t i = 0; i < gnode->get_input_size(); ++i){
            shared_ptr<descriptor::Tensor> tv = gnode->get_input_tensor_ptr(i);
            NNFUSION_CHECK_NOT_NULLPTR(tv);
            kernel_inputs.push_back(CudaKernelParameter(tv->get_name(), tv->get_shape(), tv->get_element_type(), true, i));
        }

        for (size_t i = 0; i < gnode->get_output_size(); ++i)
        {
            shared_ptr<descriptor::Tensor> tv = gnode->get_output_tensor_ptr(i);
            NNFUSION_CHECK_NOT_NULLPTR(tv);
            kernel_outputs.push_back(CudaKernelParameter(tv->get_name(), tv->get_shape(), tv->get_element_type(), false, i));
        }
    }

    // generate kernel function parameters code from kernel_inputs and kernel_outputs: half* __restrict__ input0, half* __restrict__ input1, half* __restrict__ input2, half* __restrict__ output0
    std::string generate_kernel_parameters_code(){
        std::string kernel_parameters_code = "";
        for (auto input : kernel_inputs)
        {
            kernel_parameters_code += input.get_type().c_type_string() + "* __restrict__ " + input.get_name_in_kernel() + ", ";
        }

        for (auto output : kernel_outputs)
        {
            kernel_parameters_code += output.get_type().c_type_string() + "* __restrict__ " + output.get_name_in_kernel() + ", ";
        }
        kernel_parameters_code = kernel_parameters_code.substr(0, kernel_parameters_code.size() - 2);
        return kernel_parameters_code;
    }

    //   int id = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 128;
    //   __asm__ __volatile__("prefetch.global.L2 [%0];" ::"l"((char *)prefetch0 + id) :);

    // generate prefetch code for one input tensor, according to bytes and kernel's grid_size and block_size 
    std::string generate_prefetch_code(CudaKernelParameter input){
        std::size_t prefetch_size = accumulate_prefetch_size();
        NNFUSION_CHECK(prefetch_size >= 64);
        std::size_t instruction_num = (prefetch_size - 1) / (m_total_thread_num * 128) + 1;
        
        std::string prefetch_code = "\n";
        // prefetch_code += "size_t id = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 128;\n";
        if (m_grid_size[1] * m_grid_size[2] * m_block_size[1] * m_block_size[2] == 1){
            // 1D grid of 1D blocks
            prefetch_code += "size_t id = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 128;\n";
        }else if (m_grid_size[1] * m_grid_size[2] * m_block_size[2] == 1){
            // 1D grid of 2D blocks
            prefetch_code += "size_t id = ((size_t)(blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x) * 128;\n";
        }else {
            // NNfusion only support 1D grid of 1D blocks and 1D grid of 2D blocks
            NNFUSION_CHECK_FAIL() << "NNfusion only support 1D grid of 1D blocks and 1D grid of 2D blocks";
        }
            
            
        for (size_t i = 0; i < instruction_num - 1; ++i){
            prefetch_code += "__asm__ __volatile__(\"prefetch.global.L2 [%0];\" ::\"l\"((char *)" + input.get_name_in_kernel() + " + id + " + std::to_string(i * m_total_thread_num * 128) + ") :);\n";
        }
        prefetch_code += "if (id < " + std::to_string(prefetch_size - (instruction_num - 1) * m_total_thread_num * 128) + "){\n";
        prefetch_code += "__asm__ __volatile__(\"prefetch.global.L2 [%0];\" ::\"l\"((char *)" + input.get_name_in_kernel() + " + id + " + std::to_string((instruction_num - 1) * m_total_thread_num * 128) + ") :);\n";
        prefetch_code += "}\n";

        // prefetch_code += "if (id < " + std::to_string(prefetch_size) + "){\n";
        // prefetch_code += "__asm__ __volatile__(\"prefetch.global.L2 [%0];\" ::\"l\"((char *)" + input.get_name_in_kernel() + " + id) :);\n";
        // prefetch_code += "}\n";
        return prefetch_code;
    }

    std::size_t accumulate_prefetch_size(){
        std::size_t prefetch_size = 0;
        for (size_t i = m_prefetch_begin_index; i < kernel_inputs.size(); ++i){
            size_t current_size = shape_size(kernel_inputs[i].get_shape()) * kernel_inputs[i].get_type().size();
            current_size = (current_size + 63) / 64 * 64; // align to 64 bytes, for memory pool allocation rule
            prefetch_size += current_size;
        }
        return prefetch_size;
    }

    // insert prefetch code to kernel code
    json run_prefetch(){
        auto new_kernel_parameters_code = generate_kernel_parameters_code();

        // modify kernel code(add prefetch inputs in kernel parameters): __global__ void __launch_bounds__(128) Group8(half* __restrict__ input0, half* __restrict__ input1, half* __restrict__ input2, half* __restrict__ output0)
        auto kernel_sig_start = m_code.find("__global__");
        NNFUSION_CHECK(kernel_sig_start != string::npos);
        auto kernel_sig_end = m_code.find("{", kernel_sig_start);
        NNFUSION_CHECK(kernel_sig_end != string::npos);
        auto kernel_sig = m_code.substr(kernel_sig_start, kernel_sig_end - kernel_sig_start);
                
        // split kernel_sig into kernel_sig_start, kernel_sig_params, kernel_sig_end
        auto kernel_sig_params_start = kernel_sig.rfind("(");
        NNFUSION_CHECK(kernel_sig_params_start != string::npos);
        auto kernel_sig_params_end = kernel_sig.rfind(")");
        NNFUSION_CHECK(kernel_sig_params_end != string::npos);
        NNFUSION_LOG(INFO) << "    Old kernel parameters code: " << kernel_sig.substr(kernel_sig_params_start + 1, kernel_sig_params_end - kernel_sig_params_start - 1);

        // replace kernel_sig_params with new_kernel_parameters_code
        kernel_sig = kernel_sig.substr(0, kernel_sig_params_start + 1) + new_kernel_parameters_code + kernel_sig.substr(kernel_sig_params_end);
        m_code = m_code.substr(0, kernel_sig_start) + kernel_sig + m_code.substr(kernel_sig_end);
        NNFUSION_LOG(INFO) << "    New kernel parameters code: " << new_kernel_parameters_code;

        if (m_prefetch_begin_index < kernel_inputs.size()){
            std::string prefetch_code = generate_prefetch_code(kernel_inputs[kernel_inputs.size() - 1]);
            // insert prefetch_code to kernel code: __global__ ...{... prefetch_code}
            kernel_sig_start = m_code.find("__global__");
            NNFUSION_CHECK(kernel_sig_start != string::npos);
            kernel_sig_end = m_code.find("{", kernel_sig_start);
            NNFUSION_CHECK(kernel_sig_end != string::npos);
            // find the according "}" for "{"
            int kernel_sig_end_index = kernel_sig_end;
            int kernel_sig_end_count = 1;
            while (kernel_sig_end_count > 0){
                kernel_sig_end_index++;
                if (m_code[kernel_sig_end_index] == '{')
                    kernel_sig_end_count++;
                else if (m_code[kernel_sig_end_index] == '}')
                    kernel_sig_end_count--;
            }
            m_code = m_code.substr(0, kernel_sig_end_index) + prefetch_code + m_code.substr(kernel_sig_end_index);
        }
        m_kernel_json["code"] = m_code;
        return m_kernel_json;
    }

private:
    std::shared_ptr<nnfusion::graph::GNode> m_gnode;
    json m_kernel_json;
    std::vector<CudaKernelParameter> kernel_inputs;
    std::vector<CudaKernelParameter> kernel_outputs;
    size_t m_actual_outputs;
    std::string m_code;
    size_t m_prefetch_begin_index;

    // launch config
    std::vector<size_t> m_grid_size;
    std::vector<size_t> m_block_size;
    size_t m_total_thread_num;
};

bool PrefetchPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    if (!FLAGS_fprefetch)
        return true;

    NNFUSION_LOG(INFO) << "Prefetching graph: " << graph->get_name();

    nnfusion::graph::GNodeVector kernel_node_vec;
    auto node_vec = graph->get_ordered_ops();    
    for (auto gnode : node_vec)
    {
        if (gnode->is_parameter() || gnode->is_constant() || gnode->is_variable())
            continue;
        kernel_node_vec.push_back(gnode);
    }

    NNFUSION_LOG(INFO) << "Number of nodes(ALL): " << node_vec.size();
    NNFUSION_LOG(INFO) << "Number of nodes(Kernel): " << kernel_node_vec.size();

    for (size_t index = 0; index < kernel_node_vec.size(); ++index){
        auto cur_gnode = kernel_node_vec[index];
        int cur_gnode_id = cur_gnode->get_id();
        auto cur_gnode_in_graph = graph->find_node_id(cur_gnode_id);
        NNFUSION_LOG(INFO) << "Node: " << cur_gnode_in_graph->get_name() << " (" << cur_gnode_in_graph->get_op_type() << ")";
        // for (auto input : cur_gnode_in_graph->get_in_edges())
        // {
        //     auto input_tensor = input->get_src();
        //     NNFUSION_LOG(INFO) << "    Input: " << input_tensor->get_name() << " ("
        //                        << input_tensor->get_shape() << ")" << " (" << input_tensor->get_element_type() << ")";
        // }
        // for (auto output : cur_gnode_in_graph->get_out_edges())
        // {
        //     auto output_tensor = output->get_dst();
        //     NNFUSION_LOG(INFO) << "    Output: " << output_tensor->get_name() << " ("
        //                        << output_tensor->get_shape() << ")" << " (" << output_tensor->get_element_type() << ")";
        // }
        for (size_t i = 0; i < cur_gnode_in_graph->get_input_size(); ++i) {
            auto input_tensor = cur_gnode_in_graph->get_input_tensor_ptr(i);
            NNFUSION_LOG(INFO) << "    Input: " << input_tensor->get_name() << " ("
                               << input_tensor->get_shape() << ")" << " (" << input_tensor->get_element_type() << ")";
        }
        for (size_t i = 0; i < cur_gnode_in_graph->get_output_size(); ++i) {
            auto output_tensor = cur_gnode_in_graph->get_output_tensor_ptr(i);
            NNFUSION_LOG(INFO) << "    Output: " << output_tensor->get_name() << " ("
                               << output_tensor->get_shape() << ")" << " (" << output_tensor->get_element_type() << ")";
        }
        NNFUSION_LOG(INFO) << "    Check Next Node: ";
        bool flag_prefetch = false;
        size_t prefetch_begin_index = cur_gnode_in_graph->get_input_size();
        // print next node's(in node_vec) input tensors info
        if (index < kernel_node_vec.size() - 1){
            auto next_gnode = kernel_node_vec[index + 1];

            for (auto edge : next_gnode->get_in_edges())
            {
                auto input_tensor = edge->get_src();
                if (input_tensor->is_constant()){
                    NNFUSION_LOG(INFO) << "    NextInput: " << input_tensor->get_name() << " ("
                                    << input_tensor->get_shape() << ")" << " (" << input_tensor->get_element_type() << ")";

                    int prefetch_input_id = cur_gnode_in_graph->get_input_size();
                    cur_gnode_in_graph->set_input(prefetch_input_id, next_gnode->get_inputs().at(edge->get_dst_input()));
                    auto new_edge = graph->add_edge(input_tensor, input_tensor->get_output_size() - 1, cur_gnode_in_graph, prefetch_input_id);
                    flag_prefetch = true;
                }
            }

            if (flag_prefetch && (*cur_gnode_in_graph)["Kernel_Selection_Result"].is_valid() && (*cur_gnode_in_graph)["Kernel_Selection_Result_json"].is_valid()){
                shared_ptr<KernelContext> ctx(new KernelContext(cur_gnode_in_graph));

                auto kernel = std::make_shared<CudaKernel>(cur_gnode_in_graph, (*cur_gnode_in_graph)["Kernel_Selection_Result_json"].as<json>(), prefetch_begin_index);
                auto kernel_json = kernel->run_prefetch();
                
                (*cur_gnode_in_graph)["Kernel_Selection_Result"] = std::make_pair<NNFusion_DeviceType, KernelEmitter::Pointer>(
                    nnfusion::get_device_type("CUDA_GPU"), make_shared<cuda::FusionCudaEmitter>(ctx, kernel_json));  
            }
        }
    }

    return true;
}