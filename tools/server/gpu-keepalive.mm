#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "gpu-keepalive.h"
#include <cstdio>
#include <cstdlib>
#include <future>
#include <string>

static int run_keepalive(std::atomic<bool>& stopped, std::promise<bool>& started) {
    const unsigned iterations = 500000;
    const unsigned groups = 1;
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) return 3;
        NSError * error = nil;
        NSString * source = @"#include <metal_stdlib>\nusing namespace metal;\n"
            "kernel void keepalive(device float4* out [[buffer(0)]], constant uint& n [[buffer(1)]], "
            "uint tid [[thread_position_in_grid]]) {"
            "float4 v=float4(0.1f+float(tid)*0.001f,0.2f,0.3f,0.4f);"
            "for(uint i=0;i<n;++i) { v=fma(v,float4(0.999999f),float4(0.000001f)); }"
            "out[tid]=v; }";
        id<MTLLibrary> lib = [dev newLibraryWithSource:source options:nil error:&error];
        if (!lib) { fprintf(stderr,"%s\n", error.description.UTF8String); return 4; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"keepalive"];
        id<MTLComputePipelineState> pipeline = [dev newComputePipelineStateWithFunction:fn error:&error];
        if (!pipeline) return 5;
        id<MTLCommandQueue> queue = [dev newCommandQueueWithMaxCommandBufferCount:2];
        id<MTLBuffer> output = [dev newBufferWithLength:groups*32*sizeof(float)*4 options:MTLResourceStorageModeShared];
        if (!queue || !output) return 7;
        fprintf(stderr,"keepalive: ready iterations=%u groups=%u\n", iterations,groups);
        started.set_value(true);
        uint64_t completed=0; double gpu_ms=0;
        while (!stopped.load(std::memory_order_relaxed)) {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:output offset:0 atIndex:0];
                [encoder setBytes:&iterations length:sizeof(iterations) atIndex:1];
                [encoder dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
                [encoder endEncoding]; [cb commit]; [cb waitUntilCompleted];
                if (cb.status != MTLCommandBufferStatusCompleted) {
                    fprintf(stderr,"GPU helper failed: %s\n",cb.error.description.UTF8String); return 6;
                }
                gpu_ms += (cb.GPUEndTime-cb.GPUStartTime)*1000; ++completed;
            }
        }
        fprintf(stderr,"keepalive: completed=%llu gpu_ms=%.6f mean_ms=%.6f\n",completed,gpu_ms,completed ? gpu_ms/completed : 0.0);
    }
    return 0;
}

gpu_keepalive::gpu_keepalive() {
    const char * enabled = std::getenv("LLAMA_GPU_KEEPALIVE");
    if (enabled && std::string(enabled) == "0") { ready = true; return; }
    std::promise<bool> started;
    auto status = started.get_future();
    worker = std::thread([this, started = std::move(started)]() mutable {
        const int rc = run_keepalive(stopped, started);
        if (rc) fprintf(stderr, "keepalive: worker failed (%d)\n", rc);
        try { started.set_value(false); } catch (const std::future_error&) {}
    });
    ready = status.get();
}

gpu_keepalive::~gpu_keepalive() {
    stopped.store(true, std::memory_order_relaxed);
    if (worker.joinable()) worker.join();
}
