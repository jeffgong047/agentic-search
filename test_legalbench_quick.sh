#!/bin/bash
# Quick test script for LegalBench pipeline using Docker

set -e  # Exit on error

echo "========================================================================"
echo "                  LEGALBENCH QUICK TEST (DOCKER)"
echo "========================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
ES_URL="http://elasticsearch:9200"
ES_INDEX="legalbench_test"
NUM_DOCS=50
NUM_TASKS=30
MAX_QUERIES=10

echo "${BLUE}[Step 1/5]${NC} Starting services..."
docker-compose -f docker-compose.legalbench.yml up -d

echo "${BLUE}[Step 2/5]${NC} Waiting for Elasticsearch to be healthy..."
for i in {1..30}; do
    if curl -s http://localhost:9200/_cluster/health | grep -q '"status":"green"\|"status":"yellow"'; then
        echo "${GREEN}✓ Elasticsearch is healthy${NC}"
        break
    fi
    echo "  Waiting... ($i/30)"
    sleep 2
done

echo ""
echo "${BLUE}[Step 3/5]${NC} Running download step in Docker..."
docker run --rm \
    --network agentic_search_legalbench_network \
    -v $(pwd)/legalbench:/app/legalbench \
    -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
    legalbench:latest \
    python -m legalbench download --num-docs ${NUM_DOCS}

echo ""
echo "${BLUE}[Step 4/5]${NC} Building semantic index..."
docker run --rm \
    --network agentic_search_legalbench_network \
    -v $(pwd)/legalbench:/app/legalbench \
    -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
    legalbench:latest \
    python -m legalbench build-index \
        --es-url ${ES_URL} \
        --index ${ES_INDEX} \
        --force-rebuild

echo ""
echo "${BLUE}[Step 5/5]${NC} Running evaluation (${MAX_QUERIES} queries)..."
docker run --rm \
    --network agentic_search_legalbench_network \
    -v $(pwd)/legalbench:/app/legalbench \
    -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
    legalbench:latest \
    python -m legalbench evaluate \
        --es-url ${ES_URL} \
        --index ${ES_INDEX} \
        --max-queries ${MAX_QUERIES} \
        --output /app/legalbench/results

echo ""
echo "========================================================================"
echo "${GREEN}✓ QUICK TEST COMPLETE!${NC}"
echo "========================================================================"
echo ""
echo "Results:"
echo "  - legalbench/results/results.json"
echo "  - legalbench/results/summary.md"
echo ""
echo "View report:"
echo "  cat legalbench/results/summary.md"
echo ""
