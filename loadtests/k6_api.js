import http from "k6/http";
import { check, sleep } from "k6";
import { Rate, Trend } from "k6/metrics";

// Custom metrics
const errorRate = new Rate("errors");
const forecastLatency = new Trend("forecast_latency", true);
const healthLatency = new Trend("health_latency", true);
const pipelineLatency = new Trend("pipeline_latency", true);

// Configuration
const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";
// API_KEY must be provided: k6 run -e API_KEY=your-key loadtests/k6_api.js
const API_KEY = __ENV.API_KEY;
if (!API_KEY) { throw new Error("API_KEY env var is required — use: k6 run -e API_KEY=..."); }

const headers = {
  "Content-Type": "application/json",
  "X-API-Key": API_KEY,
};

// Test stages: ramp up, sustain, spike, ramp down
export const options = {
  stages: [
    { duration: "2m", target: 10 },   // Ramp up to 10 VUs
    { duration: "5m", target: 10 },   // Sustain 10 VUs
    { duration: "1m", target: 50 },   // Spike to 50 VUs
    { duration: "3m", target: 50 },   // Sustain spike
    { duration: "1m", target: 10 },   // Ramp down
    { duration: "3m", target: 10 },   // Sustain
    { duration: "1m", target: 0 },    // Ramp down to 0
  ],
  thresholds: {
    http_req_duration: ["p(95)<500", "p(99)<1000"],
    errors: ["rate<0.01"],            // Error rate < 1%
    health_latency: ["p(95)<100"],    // Health check < 100ms P95
    forecast_latency: ["p(95)<2000"], // Forecast < 2s P95
  },
};

// --- Test scenarios ---

function testHealthCheck() {
  const res = http.get(`${BASE_URL}/api/health`);
  healthLatency.add(res.timings.duration);

  const passed = check(res, {
    "health status is 200": (r) => r.status === 200,
    "health returns ok": (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.status === "ok";
      } catch {
        return false;
      }
    },
  });

  errorRate.add(!passed);
}

function testListRuns() {
  const res = http.get(`${BASE_URL}/api/runs`, { headers });

  const passed = check(res, {
    "runs status is 200": (r) => r.status === 200,
    "runs returns array": (r) => {
      try {
        return Array.isArray(JSON.parse(r.body));
      } catch {
        return false;
      }
    },
  });

  errorRate.add(!passed);
}

function testUploadAndPipeline() {
  // Generate a small synthetic CSV payload
  const csvData =
    "date,product_id,demand,price,inventory\n" +
    "2024-01-01,SKU001,100,9.99,500\n" +
    "2024-01-02,SKU001,120,9.99,400\n" +
    "2024-01-03,SKU001,95,10.49,380\n" +
    "2024-01-04,SKU001,110,9.99,300\n" +
    "2024-01-05,SKU001,130,8.99,250\n";

  const formData = {
    file: http.file(csvData, "test_data.csv", "text/csv"),
  };

  const uploadHeaders = { "X-API-Key": API_KEY };

  const res = http.post(`${BASE_URL}/api/ingest`, formData, {
    headers: uploadHeaders,
  });

  pipelineLatency.add(res.timings.duration);

  const passed = check(res, {
    "upload status is 200 or 201": (r) =>
      r.status === 200 || r.status === 201,
  });

  errorRate.add(!passed);
}

function testGetCharts() {
  const res = http.get(`${BASE_URL}/api/runs`, { headers });

  if (res.status === 200) {
    try {
      const pipelines = JSON.parse(res.body);
      if (pipelines.length > 0) {
        const runId = pipelines[0].id;
        const chartRes = http.get(
          `${BASE_URL}/api/runs/${runId}/charts`,
          { headers }
        );

        check(chartRes, {
          "charts status is 200": (r) => r.status === 200,
        });
      }
    } catch {
      // Pipeline list may be empty
    }
  }
}

function testMetrics() {
  const res = http.get(`${BASE_URL}/api/metrics`, { headers });

  const passed = check(res, {
    "metrics status is 200": (r) => r.status === 200,
    "metrics contains prometheus data": (r) =>
      r.body && r.body.includes("chaininsight") || r.status === 404,
  });

  errorRate.add(!passed);
}

// --- Main test function ---

export default function () {
  // Weight distribution: health checks most frequent
  const rand = Math.random();

  if (rand < 0.3) {
    testHealthCheck();
  } else if (rand < 0.5) {
    testListRuns();
  } else if (rand < 0.65) {
    testGetCharts();
  } else if (rand < 0.8) {
    testMetrics();
  } else {
    testUploadAndPipeline();
  }

  sleep(Math.random() * 2 + 0.5); // 0.5-2.5s think time
}

// --- Lifecycle hooks ---

export function setup() {
  // Verify the service is reachable before starting
  const res = http.get(`${BASE_URL}/api/health`);
  if (res.status !== 200) {
    throw new Error(
      `Service not reachable at ${BASE_URL}: status ${res.status}`
    );
  }
  console.log(`Service verified at ${BASE_URL}`);
  return { startTime: new Date().toISOString() };
}

export function teardown(data) {
  console.log(`Load test completed. Started at: ${data.startTime}`);
}
