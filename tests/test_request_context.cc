// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <sstream>
#include <string>
#include <vector>

#include "mini_llama/request_context.h"
#include "tests/test_main.h"
#include "tests/test_names.h"

static bool TestRequestContextStartsWithTraceMetadata() {
  RequestContext request = StartRequest("generate", "cpu", "models/tiny");
  MINI_LLAMA_ASSERT_TRUE(request.trace_id.rfind("req_", 0) == 0);
  MINI_LLAMA_ASSERT_TRUE(request.mode == "generate");
  MINI_LLAMA_ASSERT_TRUE(request.backend == "cpu");
  MINI_LLAMA_ASSERT_TRUE(request.model_path == "models/tiny");
  MINI_LLAMA_ASSERT_TRUE(request.ok());
  return true;
}

static bool TestRequestContextRecordsEventsAndError() {
  RequestContext request = StartRequest("run", "cuda", "models/chat");
  request.RecordEvent("tokenize", 1.25, 4, "prompt");
  request.SetError("boom");
  request.Finish();

  MINI_LLAMA_ASSERT_TRUE(!request.ok());
  MINI_LLAMA_ASSERT_EQ(request.events.size(), 1);
  MINI_LLAMA_ASSERT_TRUE(request.events[0].stage == "tokenize");
  MINI_LLAMA_ASSERT_EQ(request.events[0].tokens, 4);
  MINI_LLAMA_ASSERT_TRUE(request.error == "boom");
  MINI_LLAMA_ASSERT_TRUE(request.total_ms >= 0.0);
  return true;
}

static bool TestRequestTraceFormattingIncludesStableFields() {
  RequestContext request = StartRequest("generate", "cpu", "models/tiny");
  request.prompt_tokens = 3;
  request.prefill_tokens = 3;
  request.decode_tokens = 2;
  request.generated_tokens = 2;
  request.model_load_ms = 1.0;
  request.tokenize_ms = 2.0;
  request.prefill_ms = 3.0;
  request.decode_ms = 4.0;
  request.sample_ms = 5.0;
  request.Finish();
  request.RecordEvent("prefill", 3.0, 3, "batch");

  std::string summary = FormatRequestTraceSummary(request);
  MINI_LLAMA_ASSERT_TRUE(summary.find("trace summary: trace_id=req_") !=
                         std::string::npos);
  MINI_LLAMA_ASSERT_TRUE(summary.find("status=ok") != std::string::npos);
  MINI_LLAMA_ASSERT_TRUE(summary.find("mode=generate") != std::string::npos);
  MINI_LLAMA_ASSERT_TRUE(summary.find("backend=cpu") != std::string::npos);
  MINI_LLAMA_ASSERT_TRUE(summary.find("prompt_tokens=3") != std::string::npos);
  MINI_LLAMA_ASSERT_TRUE(summary.find("prefill_ms=3.000") != std::string::npos);

  std::vector<std::string> events = FormatRequestTraceEvents(request);
  MINI_LLAMA_ASSERT_EQ(events.size(), 1);
  MINI_LLAMA_ASSERT_TRUE(events[0].find("trace event: trace_id=req_") !=
                         std::string::npos);
  MINI_LLAMA_ASSERT_TRUE(events[0].find("stage=prefill") != std::string::npos);
  MINI_LLAMA_ASSERT_TRUE(events[0].find("tokens=3") != std::string::npos);
  return true;
}

static bool TestFailedRequestKeepsTraceIdAndPrintsSummary() {
  RequestContext request = StartRequest("generate", "cpu", "models/tiny");
  const std::string trace_id = request.trace_id;
  request.SetError("Requested tokens exceed context window.");
  request.Finish();

  std::string summary = FormatRequestTraceSummary(request);
  MINI_LLAMA_ASSERT_TRUE(summary.find(trace_id) != std::string::npos);
  MINI_LLAMA_ASSERT_TRUE(summary.find("status=error") != std::string::npos);
  MINI_LLAMA_ASSERT_TRUE(
      summary.find("Requested tokens exceed context window.") !=
      std::string::npos);

  std::ostringstream out;
  PrintRequestTrace(request, out);
  MINI_LLAMA_ASSERT_TRUE(out.str().find("trace summary:") != std::string::npos);
  MINI_LLAMA_ASSERT_TRUE(out.str().find(trace_id) != std::string::npos);
  return true;
}

static bool TestPrefillEventRecordsRadixHitDetail() {
  RequestContext request = StartRequest("run", "cpu", "models/tiny");
  request.prefill_ms = 1.5;
  request.prefill_tokens = 32;
  request.RecordEvent("prefill", request.prefill_ms, request.prefill_tokens,
                      "radix_hit=60, prefix_reuse=60");
  request.Finish();

  std::vector<std::string> events = FormatRequestTraceEvents(request);
  MINI_LLAMA_ASSERT_EQ(events.size(), 1);
  MINI_LLAMA_ASSERT_TRUE(events[0].find("stage=prefill") != std::string::npos);
  MINI_LLAMA_ASSERT_TRUE(events[0].find("radix_hit=60") != std::string::npos);
  MINI_LLAMA_ASSERT_TRUE(events[0].find("prefix_reuse=60") !=
                         std::string::npos);
  return true;
}

static struct RequestContextTestRegistrar {
  RequestContextTestRegistrar() {
    RegisterTest("request_context_starts_with_trace_metadata",
                 TestRequestContextStartsWithTraceMetadata);
    RegisterTest("request_context_records_events_and_error",
                 TestRequestContextRecordsEventsAndError);
    RegisterTest("request_trace_formatting_includes_stable_fields",
                 TestRequestTraceFormattingIncludesStableFields);
    RegisterTest("failed_request_keeps_trace_id_and_prints_summary",
                 TestFailedRequestKeepsTraceIdAndPrintsSummary);
    RegisterTest("prefill_event_records_radix_hit_detail",
                 TestPrefillEventRecordsRadixHitDetail);
  }
} request_context_test_registrar;
