/**
 * \file src/opr/include/megbrain/opr/mc40_runtime_op.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include <memory>
#include "megbrain/comp_node_env.h"
#include "megbrain/graph.h"
#include "megbrain/graph/operator_node.h"
#include "megbrain/serialization/file.h"

#if MGB_MC40
#include "megbrain/mc40/mc40_memory_manager.h"

namespace mgb {
namespace opr {

MGB_DEFINE_OPR_CLASS(MC40RuntimeOpr,
                     cg::OutshapePureByInshapeOpr<cg::OperatorNodeBase>)  // {
public:
    using SharedBuffer = mgb::serialization::SharedBuffer;

    void do_execute(ExecEnv &env) override;
    void get_output_var_shape(const TensorShapeArray& inp_shape,
                              TensorShapeArray& out_shape) const override;
    void add_input_layout_constraint() override;
    void init_output_dtype() override;
    void init_output_comp_node() override;
    void on_output_comp_node_stream_changed() override;

    /**
     * \brief create MC40RuntimeOpr with buf
     */
    MC40RuntimeOpr(SharedBuffer buf, const VarNodeArray& inputs,
                   const OperatorNodeConfig& config);
    ~MC40RuntimeOpr();

    const SharedBuffer& buffer() const { return m_buffer; }

    static SymbolVarArray make(SharedBuffer buf, const SymbolVarArray& src,
                               const OperatorNodeConfig& config = {});

    static SymbolVarArray make(const void* buf, size_t size,
                               const SymbolVarArray& src,
                               const OperatorNodeConfig& config = {});

private:
    NodeProp* do_make_node_prop() const override;

    void execute_mc40();

    SharedBuffer m_buffer;
    AX_NPU_SDK_EX_HANDLE_T m_model_handle = 0;
};  // namespace opr

}  // namespace opr
}  // namespace mgb

#endif  // MGB_MC40

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
