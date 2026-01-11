#!/bin/bash
# Integration test script for p2a-mcp HTTP backend
# Run from p2a-web directory: ./scripts/test-backend.sh

set -e

API_BASE="${API_BASE:-http://localhost:8080}"
PASS=0
FAIL=0

echo "Testing p2a-mcp HTTP API at $API_BASE"
echo "========================================"
echo ""

# Helper function for tests
test_endpoint() {
    local name="$1"
    local method="$2"
    local endpoint="$3"
    local data="$4"
    local expected="$5"

    echo -n "Testing: $name... "

    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "$API_BASE$endpoint" 2>/dev/null)
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$API_BASE$endpoint" 2>/dev/null)
    fi

    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')

    if [ "$http_code" = "$expected" ]; then
        echo "PASS (HTTP $http_code)"
        ((PASS++))
        return 0
    else
        echo "FAIL (expected HTTP $expected, got $http_code)"
        echo "  Response: $body"
        ((FAIL++))
        return 1
    fi
}

# 1. Health check
test_endpoint "Health check" "GET" "/health" "" "200"

# 2. Create session
echo -n "Testing: Create session... "
session_response=$(curl -s -X POST -H "Content-Type: application/json" \
    -d '{}' "$API_BASE/api/sessions" 2>/dev/null)

if echo "$session_response" | grep -q "session_id"; then
    SESSION_ID=$(echo "$session_response" | grep -o '"session_id":"[^"]*"' | cut -d'"' -f4)
    echo "PASS (session_id: $SESSION_ID)"
    ((PASS++))
else
    echo "FAIL"
    echo "  Response: $session_response"
    ((FAIL++))
    SESSION_ID=""
fi

# Skip remaining tests if no session
if [ -z "$SESSION_ID" ]; then
    echo ""
    echo "Cannot continue without session. Exiting."
    exit 1
fi

# 3. Get session info
test_endpoint "Get session" "GET" "/api/sessions/$SESSION_ID" "" "200"

# 4. List tools
echo -n "Testing: List tools... "
tools_response=$(curl -s "$API_BASE/api/tools" 2>/dev/null)

if echo "$tools_response" | grep -q "regression_ols"; then
    tool_count=$(echo "$tools_response" | grep -o '"name"' | wc -l)
    echo "PASS ($tool_count tools found)"
    ((PASS++))
else
    echo "FAIL"
    echo "  Response: ${tools_response:0:200}..."
    ((FAIL++))
fi

# 5. Call a tool (list_datasets - should work even with no data)
echo -n "Testing: Call tool (list_datasets)... "
tool_response=$(curl -s -X POST -H "Content-Type: application/json" \
    -d "{\"session_id\":\"$SESSION_ID\",\"arguments\":{}}" \
    "$API_BASE/api/tools/list_datasets" 2>/dev/null)

if echo "$tool_response" | grep -q "success"; then
    echo "PASS"
    ((PASS++))
else
    echo "FAIL"
    echo "  Response: $tool_response"
    ((FAIL++))
fi

# 6. Call tool with data (generate_data for testing)
echo -n "Testing: Call tool (generate_data)... "
gen_response=$(curl -s -X POST -H "Content-Type: application/json" \
    -d "{\"session_id\":\"$SESSION_ID\",\"arguments\":{\"n\":100,\"columns\":[\"x\",\"y\"],\"name\":\"test_data\"}}" \
    "$API_BASE/api/tools/generate_data" 2>/dev/null)

if echo "$gen_response" | grep -q "success.*true"; then
    echo "PASS"
    ((PASS++))
else
    echo "FAIL"
    echo "  Response: $gen_response"
    ((FAIL++))
fi

# 7. Describe the generated dataset
echo -n "Testing: Call tool (describe_dataset)... "
desc_response=$(curl -s -X POST -H "Content-Type: application/json" \
    -d "{\"session_id\":\"$SESSION_ID\",\"arguments\":{\"name\":\"test_data\"}}" \
    "$API_BASE/api/tools/describe_dataset" 2>/dev/null)

if echo "$desc_response" | grep -q "success.*true"; then
    echo "PASS"
    ((PASS++))
else
    echo "FAIL"
    echo "  Response: $desc_response"
    ((FAIL++))
fi

# 8. Run OLS regression
echo -n "Testing: Call tool (regression_ols)... "
ols_response=$(curl -s -X POST -H "Content-Type: application/json" \
    -d "{\"session_id\":\"$SESSION_ID\",\"arguments\":{\"dataset\":\"test_data\",\"y\":\"y\",\"x\":[\"x\"],\"intercept\":true}}" \
    "$API_BASE/api/tools/regression_ols" 2>/dev/null)

if echo "$ols_response" | grep -q "success.*true"; then
    echo "PASS"
    ((PASS++))
else
    echo "FAIL"
    echo "  Response: $ols_response"
    ((FAIL++))
fi

# 9. LLM models endpoint (may fail if Ollama not running)
echo -n "Testing: LLM models... "
models_response=$(curl -s "$API_BASE/api/llm/models" 2>/dev/null)
http_code=$(curl -s -o /dev/null -w "%{http_code}" "$API_BASE/api/llm/models" 2>/dev/null)

if [ "$http_code" = "200" ]; then
    echo "PASS"
    ((PASS++))
else
    echo "SKIP (Ollama may not be running)"
fi

# 10. Delete session
test_endpoint "Delete session" "DELETE" "/api/sessions/$SESSION_ID" "" "200"

# Summary
echo ""
echo "========================================"
echo "Results: $PASS passed, $FAIL failed"
echo ""

if [ $FAIL -gt 0 ]; then
    exit 1
else
    echo "All tests passed!"
    exit 0
fi
