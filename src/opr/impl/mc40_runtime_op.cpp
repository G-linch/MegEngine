/**
 * \file src/opr/impl/mc40_runtime_op.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/opr/mc40_runtime_op.h"
#include "megbrain/common.h"
#include "megbrain/graph/event.h"
#include "megdnn/dtype.h"

#include <memory>

#if MGB_MC40

using namespace mgb;
using namespace opr;

namespace {
TensorShape mc40_shape_to_mgb_shape(AX_NPU_SDK_EX_TENSOR_META_T tensor_meta) {
    TensorShape ret;
    ret.ndim = tensor_meta.nShapeNDim;
    for (size_t i = 0; i < ret.ndim; ++i) {
        ret[i] = tensor_meta.pShape[i];
    }
    return ret;
}
DType mc40_dtype_to_mgb_dtype(AX_NPU_SDK_EX_ADV_TENSOR_DTYPE data_type) {
    switch (data_type) {
        case AX_NPU_TDT_UINT8:
            return dtype::Uint8();
        case AX_NPU_TDT_FLOAT32:
            return dtype::Float32();
        case AX_NPU_TDT_INT16:
            return dtype::Int16();
        case AX_NPU_TDT_INT32:
            return dtype::Int32();
        default:
            mgb_throw(MegBrainError,
                      "MC40DataType %d is not supported by MegBrain.",
                      static_cast<int>(data_type));
    }
}

};  // namespace

/* ====================== MC40RuntimeOpr ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(MC40RuntimeOpr);
MC40RuntimeOpr::MC40RuntimeOpr(SharedBuffer buf, const VarNodeArray& inputs,
                               const OperatorNodeConfig& config)
        : Super(inputs[0]->owner_graph(), config, "mc40_runtime", inputs),
          m_buffer{std::move(buf)} {
    mgb_assert(
            inputs[0]->comp_node().device_type() == CompNode::DeviceType::MC40,
            "MC40RuntimeOpr can only be used on mc40 comp node; "
            "got %s",
            inputs[0]->comp_node().to_string().c_str());

    for (auto i : inputs) {
        add_input({i});
    }
    MGB_MC40_CHECK(AX_NPU_SDK_EX_Create_handle(&m_model_handle, m_buffer.data(),
                                               m_buffer.size()));

    const AX_NPU_SDK_EX_ADV_IO_INFO_T* io_info =
            AX_NPU_SDK_EX_ADV_Get_io_info(m_model_handle);

    size_t nr_outputs = io_info->nOutputSize;
    bool has_workspace = false;
    if (nr_outputs == 1) {
        add_output(None);
        const auto& tensor_meta = *(io_info->pOutputs[0].pTensorMeta);
        if (tensor_meta.eMemoryType == AX_NPU_MT_VIRTUAL) {
            mgb_assert(tensor_meta.nInnerSize > 0);
            has_workspace = true;
        }

    } else {
        for (size_t i = 0; i < nr_outputs; ++i) {
            add_output(ssprintf("o%zu", i));

            const auto& tensor_meta = *(io_info->pOutputs[i].pTensorMeta);
            if (tensor_meta.eMemoryType == AX_NPU_MT_VIRTUAL) {
                mgb_assert(tensor_meta.nInnerSize > 0);
                has_workspace = true;
            }
        }
    }
    mgb_assert(has_workspace, "Currently only support model with cpu tail");

    //! As we may directly forward memory of the output to dev to reduce the
    //! overhead of memory copy, so here we use dynamic memory alloc.
    using F = VarNode::Flag;
    for (size_t i = 0; i < nr_outputs; ++i) {
        output(i)->add_flag(F::NO_SYS_MEM_ALLOC).add_flag(F::NO_MEM_RECLAIM);
    }

    add_equivalence_component<mgb::ScalarHash<const void*>>(m_buffer.data());
    cg::add_workspace_output(this);
};

MC40RuntimeOpr::~MC40RuntimeOpr() {
    MGB_MC40_CHECK(AX_NPU_SDK_EX_Destroy_handle(m_model_handle));
}

void MC40RuntimeOpr::execute_mc40() {
    auto&& mc40_env =
            CompNodeEnv::from_comp_node(input(0)->comp_node()).mc40_env();
    mc40_env.activate();

    const AX_NPU_SDK_EX_ADV_IO_INFO_T* io_info =
            AX_NPU_SDK_EX_ADV_Get_io_info(m_model_handle);

    for (size_t i = 0; i < io_info->nOutputSize; i++) {
        auto ovar = output(i);
        ovar->shape_alloc(ovar->shape());
    }

    AX_NPU_SDK_EX_IO_T npu_io;
    memset(&npu_io, 0, sizeof(npu_io));
    size_t batch_size = input(0)->dev_tensor().layout().shape[0];
    size_t model_batch = (*(io_info->pInputs[0].pTensorMeta)).pShape[0];
    for (size_t batch_idx = 0; batch_idx < batch_size;
         batch_idx += model_batch) {
        //! prepare input
        npu_io.nInputSize = io_info->nInputSize;
        auto inputs =
                std::make_unique<AX_NPU_SDK_EX_BUF_T[]>(npu_io.nInputSize);
        npu_io.pInputs = inputs.get();
        for (size_t i = 0; i < npu_io.nInputSize; i++) {
            // get input addr info
            AX_VOID* p_virtual_addr = input(i)->dev_tensor().raw_ptr();
            AX_U64 phy_addr =
                    MC40MemoryManager::Instance().get_phyaddr(p_virtual_addr);
            auto nr_bytes_per_batch =
                    input(i)->layout().span().dist_byte() / batch_size;
            // add batch offset
            p_virtual_addr = reinterpret_cast<AX_VOID*>(
                    reinterpret_cast<AX_U64>(p_virtual_addr) +
                    nr_bytes_per_batch * batch_idx);
            phy_addr += nr_bytes_per_batch * batch_idx;

            MGB_MC40_CHECK(AX_NPU_SDK_EX_ADV_Make_io_buffer(
                    phy_addr, p_virtual_addr, nr_bytes_per_batch, phy_addr,
                    p_virtual_addr, nr_bytes_per_batch, &npu_io.pInputs[i]));
        }

        //! prepare output
        npu_io.nOutputSize = io_info->nOutputSize;
        auto outputs =
                std::make_unique<AX_NPU_SDK_EX_BUF_T[]>(npu_io.nOutputSize);
        npu_io.pOutputs = outputs.get();
        AX_U32 offset = 0;
        AX_VOID* inner_virtual_addr_start = nullptr;
        AX_U64 inner_phy_addr_start = 0;
        // get innder addr form workspace
        inner_virtual_addr_start =
                output(npu_io.nOutputSize)->dev_tensor().raw_ptr();
        inner_phy_addr_start = MC40MemoryManager::Instance().get_phyaddr(
                inner_virtual_addr_start);
        for (size_t i = 0; i < npu_io.nOutputSize; i++) {
            // get output addr info
            AX_VOID* p_virtual_addr = output(i)->dev_tensor().raw_ptr();
            AX_U64 phy_addr = 0;
            auto nr_bytes_per_batch =
                    output(i)->layout().span().dist_byte() / batch_size;
            // add batch offset
            p_virtual_addr = reinterpret_cast<AX_VOID*>(
                    reinterpret_cast<AX_U64>(p_virtual_addr) +
                    nr_bytes_per_batch * batch_idx);
            phy_addr += nr_bytes_per_batch * batch_idx;

            const auto& tensor_meta = *(io_info->pOutputs[i].pTensorMeta);
            if (tensor_meta.eMemoryType == AX_NPU_MT_PHYSICAL) {
                MGB_MC40_CHECK(AX_NPU_SDK_EX_ADV_Make_io_buffer(
                        phy_addr, p_virtual_addr, nr_bytes_per_batch, phy_addr,
                        p_virtual_addr, nr_bytes_per_batch,
                        &npu_io.pOutputs[i]));
            } else if (tensor_meta.eMemoryType == AX_NPU_MT_VIRTUAL) {
                auto p_inner_virtual_addr = reinterpret_cast<AX_VOID*>(
                        reinterpret_cast<AX_U64>(inner_virtual_addr_start) +
                        offset);
                auto innerphy_addr = inner_phy_addr_start + offset;
                MGB_MC40_CHECK(AX_NPU_SDK_EX_ADV_Make_io_buffer(
                        phy_addr, p_virtual_addr, nr_bytes_per_batch,
                        innerphy_addr, p_inner_virtual_addr,
                        tensor_meta.nInnerSize, &npu_io.pOutputs[i]));

                offset += tensor_meta.nInnerSize;
            }
        }

        // execute model task
        MGB_MC40_CHECK(AX_NPU_SDK_EX_Run_task_sync(m_model_handle, &npu_io));
    }
}

void MC40RuntimeOpr::init_output_comp_node() {
    //! set output to cpu compnode if has cpu tail
    const AX_NPU_SDK_EX_ADV_IO_INFO_T* io_info =
            AX_NPU_SDK_EX_ADV_Get_io_info(m_model_handle);

    CompNode input_cn;
    for (auto&& i : input()) {
        if (!input_cn.valid()) {
            input_cn = i->comp_node();
        } else {
            mgb_assert(input_cn.mem_node() == i->comp_node().mem_node(),
                       "opr %s{%s} requires all input to be on the same memory "
                       "node expect=%s cur_var=%s cur_cn=%s",
                       this->cname(), this->dyn_typeinfo()->name,
                       input_cn.to_string().c_str(), i->cname(),
                       i->comp_node().to_string().c_str());
        }
    }
    for (size_t i = 0; i < io_info->nOutputSize; i++) {
        //! compnode of the var should be default_cpu as the output will be
        //! proxy to user
        output(i)->comp_node(CompNode::default_cpu());
    }
    //! the last output is workspace, which should be the same as input
    output(io_info->nOutputSize)->comp_node(input_cn);
}

MC40RuntimeOpr::NodeProp* MC40RuntimeOpr::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    ret->add_flag(NodeProp::Flag::CROSS_COMP_NODE_MEMORY);
    return ret;
}

void MC40RuntimeOpr::do_execute(ExecEnv& env) {
    CompNode cn = output(0)->comp_node();
    auto runner = [this, cn]() {
        this->owner_graph()->event().signal_inplace<cg::event::BeforeKernel>(
                this, cn);
        cn.activate();
        execute_mc40();
        this->owner_graph()->event().signal_inplace<cg::event::AfterKernel>(
                this, cn);
    };
    env.dispatch_on_comp_node(cn, runner);

    // Send BeforeKernel/AfterKernel event on every different comp_node
    ThinHashSet<mgb::CompNode> st = cg::get_opr_comp_node_set(this);
    for (auto cn : st) {
        auto send_event = [this, cn]() {
            this->owner_graph()
                    ->event()
                    .signal_inplace<cg::event::BeforeKernel>(this, cn);
            this->owner_graph()->event().signal_inplace<cg::event::AfterKernel>(
                    this, cn);
        };
        env.dispatch_on_comp_node(cn, send_event);
    }
}

void MC40RuntimeOpr::on_output_comp_node_stream_changed() {
    mgb_throw(SystemError, "comp node of output should not change");
}

void MC40RuntimeOpr::get_output_var_shape(const TensorShapeArray& inp_shape,
                                          TensorShapeArray& out_shape) const {
    const AX_NPU_SDK_EX_ADV_IO_INFO_T* io_info =
            AX_NPU_SDK_EX_ADV_Get_io_info(m_model_handle);
    size_t nr_inputs = io_info->nInputSize;

    for (size_t i = 0; i < nr_inputs; ++i) {
        const auto& tensor_meta = *(io_info->pInputs[i].pTensorMeta);
        auto model_shape = mc40_shape_to_mgb_shape(tensor_meta);
        // enable mutibatch
        mgb_assert(inp_shape[i][0] % model_shape[0] == 0,
                   "input %zu batch is %zu, while model's input batch is %zu",
                   i, inp_shape[i][0], model_shape[0]);
        model_shape[0] = inp_shape[i][0];
        mgb_assert(model_shape.eq_shape(inp_shape[i]),
                   "shape mismatch of input %zu, expected: %s got: %s", i,
                   model_shape.to_string().c_str(),
                   inp_shape[i].to_string().c_str());
    }
    size_t input_batch = (*(io_info->pInputs[0].pTensorMeta)).pShape[0];
    AX_U32 workspace_size = 0;
    for (size_t i = 0; i < io_info->nOutputSize; ++i) {
        const auto& tensor_meta = *(io_info->pOutputs[i].pTensorMeta);
        out_shape[i] = mc40_shape_to_mgb_shape(tensor_meta);
        // enable mutibatch
        out_shape[i][0] = out_shape[i][0] * inp_shape[0][0] / input_batch;
        if (tensor_meta.eMemoryType == AX_NPU_MT_VIRTUAL) {
            workspace_size += tensor_meta.nInnerSize;
        }
    }
    out_shape.back() = {workspace_size};
}

void MC40RuntimeOpr::add_input_layout_constraint() {
    //! default contiguous
    for (auto i : input()) {
        i->add_layout_constraint_contiguous();
    }
}

void MC40RuntimeOpr::init_output_dtype() {
    DType dt_mc40, dt_input;
    const AX_NPU_SDK_EX_ADV_IO_INFO_T* io_info =
            AX_NPU_SDK_EX_ADV_Get_io_info(m_model_handle);

    for (size_t i = 0; i < io_info->nInputSize; ++i) {
        dt_mc40 = mc40_dtype_to_mgb_dtype(io_info->pInputs[i].eDType);
        dt_input = input(i)->dtype();
        mgb_assert(dt_mc40.valid() && dt_input.valid() &&
                           dt_mc40.enumv() == dt_input.enumv(),
                   "dtype mismatch of input %zu: expected %s, "
                   "got %s",
                   i, dt_mc40.name(), dt_input.name());
    }

    for (size_t i = 0; i < io_info->nOutputSize; ++i) {
        dt_mc40 = mc40_dtype_to_mgb_dtype(io_info->pOutputs[i].eDType);
        mgb_assert(dt_mc40.valid(),
                   "output dtype checking failed: invalid dtype returned.");
        if (!output(i)->dtype().valid())
            output(i)->dtype(dt_mc40);
    }
}

SymbolVarArray MC40RuntimeOpr::make(SharedBuffer buf, const SymbolVarArray& src,
                                    const OperatorNodeConfig& config) {
    VarNodeArray var_node_array = cg::to_var_node_array(src);
    auto mc40_runtime_opr = std::make_unique<MC40RuntimeOpr>(
            std::move(buf), var_node_array, config);
    auto ret = cg::to_symbol_var_array(
            src[0].node()
                    ->owner_graph()
                    ->insert_opr(std::move(mc40_runtime_opr))
                    ->output());
    ret.pop_back();  // remove workspace
    return ret;
}

SymbolVarArray MC40RuntimeOpr::make(const void* buf, size_t size,
                                    const SymbolVarArray& src,
                                    const OperatorNodeConfig& config) {
    mgb_throw_if(!CompNode::get_device_count(CompNode::DeviceType::MC40),
                 SystemError,
                 "can not create MC40RuntimeOpr when mc40 is not "
                 "available");
    std::shared_ptr<uint8_t> shptr{new uint8_t[size],
                                   [](uint8_t* p) { delete[] p; }};
    memcpy(shptr.get(), buf, size);
    SharedBuffer buffer{std::move(shptr), size};
    return make(std::move(buffer), src, config);
}

#endif  // MGB_MC40

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
