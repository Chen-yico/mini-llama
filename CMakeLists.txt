cmake_minimum_required(VERSION 3.17)

project(mini_llama LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
)

add_executable(mini-llama
  ${MINI_LLAMA_CORE_SOURCES}
  src/main.cc
)
set_target_properties(mini-llama PROPERTIES OUTPUT_NAME mini-llama)

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
)

add_executable(mini-llama-tests ${TEST_SOURCES})
set_target_properties(mini-llama-tests PROPERTIES OUTPUT_NAME mini-llama-tests)

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
endif()
