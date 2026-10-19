cmake_minimum_required(VERSION 3.17)

project(mini_llama LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MINI_LLAMA_CUDA "Build NVIDIA CUDA backend" OFF)
if(MINI_LLAMA_CUDA)
  enable_language(CUDA)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
  endif()
  find_package(CUDAToolkit REQUIRED)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR})

set(MINI_LLAMA_CORE_SOURCES
  src/tensor.cc
  src/ops.cc
  src/thread_pool.cc
  src/matmul_optimized.cc
  src/tokenizer.cc
  src/json_vocab_tokenizer.cc
  src/bpe_tokenizer.cc
  src/gguf.cc
  src/gguf_loader.cc
  src/gguf_tokenizer.cc
  src/quant.cc
  src/quantized_tensor.cc
  src/kv_cache.cc
  src/context.cc
  src/batch.cc
  src/radix_tree.cc
  src/request_context.cc
  src/chat.cc
  src/prompt_builder.cc
  src/terminal.cc
  src/model.cc
  src/loader.cc
  src/forward.cc
  src/sampler.cc
  src/debug.cc
  src/backend.cc
  src/cuda_runtime.cc
  src/cuda_tensor.cc
  src/cuda_matmul.cc
  src/cuda_quant.cc
  src/cuda_ops.cc
  src/cuda_kv_cache.cc
  src/cuda_attention.cc
  src/cuda_weights.cc
  src/forward_cuda.cc
)
if(MINI_LLAMA_CUDA)
  list(APPEND MINI_LLAMA_CORE_SOURCES
    src/cuda_attention.cu
    src/cuda_ops.cu
    src/cuda_quant.cu
  )
endif()

add_executable(mini-llama
  ${MINI_LLAMA_CORE_SOURCES}
  src/main.cc
)
set_target_properties(mini-llama PROPERTIES OUTPUT_NAME mini-llama)
if(MINI_LLAMA_CUDA)
  target_compile_definitions(mini-llama PRIVATE MINI_LLAMA_USE_CUDA=1)
  target_link_libraries(mini-llama PRIVATE CUDA::cudart CUDA::cublas)
endif()

set(TEST_SOURCES
  ${MINI_LLAMA_CORE_SOURCES}
  tests/test_main.cc
  tests/test_tensor.cc
  tests/test_ops.cc
  tests/test_tokenizer.cc
  tests/test_gguf_tokenizer.cc
  tests/test_kv_cache.cc
  tests/test_batch.cc
  tests/test_radix_tree.cc
  tests/test_request_context.cc
  tests/test_chat.cc
  tests/test_forward.cc
  tests/test_sampler.cc
  tests/test_loader.cc
  tests/test_gguf.cc
  tests/test_gguf_loader.cc
  tests/test_debug.cc
  tests/test_quant.cc
  tests/test_threaded_matmul.cc
)

add_executable(mini-llama-tests ${TEST_SOURCES})
set_target_properties(mini-llama-tests PROPERTIES OUTPUT_NAME mini-llama-tests)
if(MINI_LLAMA_CUDA)
  target_compile_definitions(mini-llama-tests PRIVATE MINI_LLAMA_USE_CUDA=1)
  target_link_libraries(mini-llama-tests PRIVATE CUDA::cudart CUDA::cublas)
endif()

enable_testing()
add_test(NAME mini-llama-tests COMMAND mini-llama-tests
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(
    NAME cli-smoke
    COMMAND ${CMAKE_COMMAND} -E env
            MINI_LLAMA_BIN=$<TARGET_FILE:mini-llama>
            MINI_LLAMA_SKIP_REAL_MODEL=1
            ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/scripts/test_real_model_smoke.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
  add_test(
    NAME threaded-cli-smoke
    COMMAND ${CMAKE_COMMAND} -E env
            MINI_LLAMA_BIN=$<TARGET_FILE:mini-llama>
            ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/scripts/test_threaded_cli.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
  add_test(
    NAME golden-logits
    COMMAND ${CMAKE_COMMAND} -E env
            MINI_LLAMA_BIN=$<TARGET_FILE:mini-llama>
            ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/scripts/test_golden.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
  add_test(
    NAME backend-cli-smoke
    COMMAND ${CMAKE_COMMAND} -E env
            MINI_LLAMA_BIN=$<TARGET_FILE:mini-llama>
            ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/scripts/test_backend_cli.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
  add_test(
    NAME cuda-chat-smoke
    COMMAND ${CMAKE_COMMAND} -E env
            MINI_LLAMA_BIN=$<TARGET_FILE:mini-llama>
            ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/scripts/test_cuda_chat_smoke.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
endif()

function(mini_llama_cuda_test target)
  add_executable(${target} ${ARGN})
  set_target_properties(${target} PROPERTIES OUTPUT_NAME ${target})
  if(MINI_LLAMA_CUDA)
    target_compile_definitions(${target} PRIVATE MINI_LLAMA_USE_CUDA=1)
    target_link_libraries(${target} PRIVATE CUDA::cudart CUDA::cublas)
  endif()
endfunction()

mini_llama_cuda_test(mini-llama-cuda-runtime-tests
  src/cuda_runtime.cc
  tests/test_cuda_runtime.cc)
add_test(NAME cuda_runtime COMMAND mini-llama-cuda-runtime-tests
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

mini_llama_cuda_test(mini-llama-cuda-matmul-tests
  src/tensor.cc
  src/cuda_runtime.cc
  src/cuda_tensor.cc
  src/cuda_matmul.cc
  tests/test_cuda_matmul.cc)
add_test(NAME cuda_matmul COMMAND mini-llama-cuda-matmul-tests matmul
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME cuda_linear COMMAND mini-llama-cuda-matmul-tests linear
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

mini_llama_cuda_test(mini-llama-cuda-quant-tests
  ${MINI_LLAMA_CORE_SOURCES}
  tests/test_cuda_quant.cc)
add_test(NAME cuda_quant_q8_0 COMMAND mini-llama-cuda-quant-tests q8_0
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME cuda_quant_q4 COMMAND mini-llama-cuda-quant-tests q4
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME cuda_forward_quant COMMAND mini-llama-cuda-quant-tests forward
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME cuda_forward_q4 COMMAND mini-llama-cuda-quant-tests forward_q4
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

mini_llama_cuda_test(mini-llama-cuda-forward-buffer-tests
  ${MINI_LLAMA_CORE_SOURCES}
  tests/test_cuda_forward_buffer.cc)
add_test(NAME cuda_forward_buffer COMMAND mini-llama-cuda-forward-buffer-tests f32
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME cuda_forward_device_resident
         COMMAND mini-llama-cuda-forward-buffer-tests q8_0
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

mini_llama_cuda_test(mini-llama-cuda-attention-tests
  ${MINI_LLAMA_CORE_SOURCES}
  tests/test_cuda_attention.cc)
add_test(NAME cuda_attention COMMAND mini-llama-cuda-attention-tests direct
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME cuda_kv_attention COMMAND mini-llama-cuda-attention-tests kv
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME cuda_forward_attention
         COMMAND mini-llama-cuda-attention-tests forward
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

mini_llama_cuda_test(mini-llama-cuda-forward-linear-tests
  ${MINI_LLAMA_CORE_SOURCES}
  tests/test_cuda_forward_linear.cc)
add_test(NAME cuda_forward_linear COMMAND mini-llama-cuda-forward-linear-tests
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

mini_llama_cuda_test(mini-llama-cuda-ops-tests
  ${MINI_LLAMA_CORE_SOURCES}
  tests/test_cuda_ops.cc)
add_test(NAME cuda_ops COMMAND mini-llama-cuda-ops-tests
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

mini_llama_cuda_test(mini-llama-cuda-kv-cache-tests
  ${MINI_LLAMA_CORE_SOURCES}
  tests/test_cuda_kv_cache.cc)
add_test(NAME cuda_kv_cache COMMAND mini-llama-cuda-kv-cache-tests
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

mini_llama_cuda_test(mini-llama-cuda-forward-tests
  ${MINI_LLAMA_CORE_SOURCES}
  tests/test_cuda_forward.cc)
add_test(NAME cuda_forward COMMAND mini-llama-cuda-forward-tests
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
