/**
 * \file src/opr/test/mc40_runtime_op.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain_build_config.h"

#if MGB_MC40

#include "megbrain/comp_node_env.h"
#include "megbrain/opr/io.h"
#include "megbrain/test/helper.h"
#include "megbrain/opr/mc40_runtime_op.h"
#include "megbrain/plugin/profiler.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "./mc40_models.h"

using namespace mgb;
using namespace opr;

#if MEGDNN_AARCH64
TEST(TestOprMC40, RuntimeBasic) {
    for (size_t batch : {1, 4, 8}) {
        HostTensorGenerator<dtype::Uint8> gen;
        std::shared_ptr<HostTensorND> host_x = gen({batch, 192, 128, 1});
        auto graph = ComputingGraph::make();

        //! run neu model
        const auto& neu_buffer = MC40_MODEL.at("sinopec_nv12_extra_neu");
        auto cn = CompNode::load("mc40:0");
        auto x = Host2DeviceCopy::make(*graph, host_x, {cn});
        auto y = opr::MC40RuntimeOpr::make(neu_buffer.first, neu_buffer.second,
                                           {x})[0];

        ASSERT_EQ(y.node()->comp_node().device_type(),
                  CompNode::DeviceType::CPU);
        HostTensorND host_y, host_y_proxy;
        auto func =
                graph->compile({make_callback_copy(y, host_y, true),
                                {y, [&host_y_proxy](DeviceTensorND& d) {
                                     host_y_proxy =
                                             mgb::HostTensorND::make_proxy(d);
                                 }}});
        func->execute().wait();
        MGB_ASSERT_TENSOR_NEAR(host_y, host_y_proxy, 1e-4);
    }
}

TEST(TestOprMC40, Profiling) {
    HostTensorGenerator<dtype::Uint8> gen;
    const auto& graph = ComputingGraph::make();
    GraphProfiler profiler{graph.get()};
    std::shared_ptr<HostTensorND> host_x = gen({1, 192, 128, 1});

    const auto& neu_buffer = MC40_MODEL.at("sinopec_nv12_extra_neu");
    auto cn = CompNode::load("mc40:0");
    auto x = Host2DeviceCopy::make(*graph, host_x, cn);
    auto y = opr::MC40RuntimeOpr::make(neu_buffer.first, neu_buffer.second,
                                       {x})[0];
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y, true)});
    func->execute().wait();

    profiler.to_json_full(func.get())
            ->writeto_fpath(output_file("mc40_runtime_opr_profile.json"));
}

#endif

#endif  // MGB_MC40

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
