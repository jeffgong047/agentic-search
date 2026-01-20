#!/bin/bash
# Quick runner for trace debugging tests with Claude Opus

set -e  # Exit on error

echo "========================================================================="
echo "                 AGENTIC SEARCH - TRACE DEBUGGING TESTS"
echo "========================================================================="
echo ""

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "❌ ERROR: ANTHROPIC_API_KEY not set"
    echo ""
    echo "Get your key from: https://console.anthropic.com/"
    echo ""
    echo "Then run:"
    echo "  export ANTHROPIC_API_KEY='sk-ant-...'"
    echo "  bash run_trace_tests.sh"
    echo ""
    exit 1
fi

echo "✓ ANTHROPIC_API_KEY is set"
echo "✓ Model: Claude Opus 4.5"
echo ""
echo "========================================================================="
echo ""

# Run the tests
python test_trace_debug.py

echo ""
echo "========================================================================="
echo "                              COMPLETE"
echo "========================================================================="
echo ""
echo "Next steps:"
echo "  1. Review traces above to understand agent behavior"
echo "  2. If tests fail, check specific error messages"
echo "  3. Read TESTING_TRACE_GUIDE.md for detailed explanations"
echo ""
