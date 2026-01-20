#!/usr/bin/env python3
"""
Agent Trace Analysis - Data Science Toolkit
Industry best practices for analyzing LLM agent behavior

References:
- LangSmith (LangChain): Trace visualization & debugging
- Phoenix (Arize AI): LLM observability platform  
- Weights & Biases: Agent tracking & analysis
- OpenTelemetry: Distributed trace standards
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import numpy as np

def load_results(results_path="legalbench/results/results.json"):
    """Load evaluation results"""
    with open(results_path) as f:
        return json.load(f)

def analyze_latency_distribution(data):
    """Analyze latency patterns - identify fast vs slow queries"""
    queries = data['per_query_results']
    latencies = [q['latency_ms'] for q in queries]
    
    # Identify bimodal distribution
    fast_threshold = 1000  # 1 second
    fast_queries = [l for l in latencies if l < fast_threshold]
    slow_queries = [l for l in latencies if l >= fast_threshold]
    
    print("=== LATENCY ANALYSIS ===")
    print(f"Total queries: {len(latencies)}")
    print(f"Fast queries (<1s): {len(fast_queries)} ({len(fast_queries)/len(latencies)*100:.1f}%)")
    print(f"  Mean: {np.mean(fast_queries):.0f}ms, Median: {np.median(fast_queries):.0f}ms")
    print(f"Slow queries (≥1s): {len(slow_queries)} ({len(slow_queries)/len(latencies)*100:.1f}%)")
    if slow_queries:
        print(f"  Mean: {np.mean(slow_queries):.0f}ms, Median: {np.median(slow_queries):.0f}ms")
    print()
    
    return {
        'fast': fast_queries,
        'slow': slow_queries,
        'all': latencies
    }

def analyze_retrieval_success(data):
    """Analyze retrieval patterns"""
    queries = data['per_query_results']
    
    retrieval_counts = [q.get('num_retrieved', 0) for q in queries]
    zero_results = sum(1 for c in retrieval_counts if c == 0)
    partial_results = sum(1 for c in retrieval_counts if 0 < c < 20)
    full_results = sum(1 for c in retrieval_counts if c == 20)
    
    print("=== RETRIEVAL SUCCESS ANALYSIS ===")
    print(f"Zero results: {zero_results}/{len(queries)} ({zero_results/len(queries)*100:.1f}%)")
    print(f"Partial results (1-19): {partial_results}/{len(queries)} ({partial_results/len(queries)*100:.1f}%)")
    print(f"Full results (20): {full_results}/{len(queries)} ({full_results/len(queries)*100:.1f}%)")
    print(f"Mean docs retrieved: {np.mean(retrieval_counts):.1f}")
    print()
    
    return {
        'zero': zero_results,
        'partial': partial_results,
        'full': full_results,
        'counts': retrieval_counts
    }

def analyze_domain_coverage(data):
    """Analyze query distribution across legal domains"""
    queries = data['per_query_results']
    
    domain_counts = Counter(q['domain'] for q in queries)
    category_counts = Counter(q['category'] for q in queries)
    
    print("=== DOMAIN COVERAGE ===")
    for domain, count in domain_counts.most_common():
        print(f"{domain:20s}: {count:3d} queries")
    print()
    
    print("=== CATEGORY COVERAGE ===")
    for category, count in category_counts.most_common():
        print(f"{category:20s}: {count:3d} queries")
    print()
    
    return domain_counts, category_counts

def create_latency_plot(latency_data, output_path="legalbench/results/latency_distribution.png"):
    """Create latency distribution visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Agent Latency Analysis', fontsize=16, fontweight='bold')
    
    all_latencies = latency_data['all']
    
    # 1. Histogram with bimodal highlighting
    ax = axes[0, 0]
    ax.hist(latency_data['fast'], bins=20, alpha=0.7, label='Fast (<1s)', color='green')
    ax.hist(latency_data['slow'], bins=20, alpha=0.7, label='Slow (≥1s)', color='red')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Latency Distribution (Bimodal Pattern)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. CDF plot
    ax = axes[0, 1]
    sorted_latencies = np.sort(all_latencies)
    cdf = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)
    ax.plot(sorted_latencies, cdf, linewidth=2, color='blue')
    ax.axhline(0.5, color='gray', linestyle='--', label='Median')
    ax.axhline(0.95, color='red', linestyle='--', label='P95')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Function')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Box plot comparison
    ax = axes[1, 0]
    box_data = [latency_data['fast'], latency_data['slow']] if latency_data['slow'] else [latency_data['fast']]
    labels = ['Fast\n(<1s)', 'Slow\n(≥1s)'] if latency_data['slow'] else ['Fast\n(<1s)']
    box = ax.boxplot(box_data, labels=labels, patch_artist=True)
    for patch, color in zip(box['boxes'], ['lightgreen', 'lightcoral'][:len(box_data)]):
        patch.set_facecolor(color)
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Fast vs Slow Query Comparison')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Percentiles table
    ax = axes[1, 1]
    ax.axis('off')
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    p_values = [np.percentile(all_latencies, p) for p in percentiles]
    
    table_data = [[f'P{p}', f'{v:.0f} ms'] for p, v in zip(percentiles, p_values)]
    table = ax.table(cellText=table_data, colLabels=['Percentile', 'Latency'],
                     cellLoc='center', loc='center', colWidths=[0.4, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax.set_title('Latency Percentiles', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved latency visualization: {output_path}")
    return output_path

def create_retrieval_plot(retrieval_data, output_path="legalbench/results/retrieval_analysis.png"):
    """Create retrieval success visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Retrieval Success Analysis', fontsize=16, fontweight='bold')
    
    # 1. Pie chart of retrieval outcomes
    ax = axes[0]
    sizes = [retrieval_data['zero'], retrieval_data['partial'], retrieval_data['full']]
    labels = [f'Zero Results\n({retrieval_data["zero"]})', 
              f'Partial\n({retrieval_data["partial"]})',
              f'Full (20)\n({retrieval_data["full"]})']
    colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
    explode = (0.1, 0, 0)  # Highlight zero results
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90, textprops={'fontsize': 11})
    ax.set_title('Retrieval Outcome Distribution')
    
    # 2. Histogram of retrieval counts
    ax = axes[1]
    counts = retrieval_data['counts']
    ax.hist(counts, bins=21, range=(-0.5, 20.5), color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(counts), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(counts):.1f}')
    ax.set_xlabel('Number of Documents Retrieved')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Retrieval Counts')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved retrieval visualization: {output_path}")
    return output_path

def main():
    """Run complete trace analysis"""
    print("=" * 60)
    print("AGENT TRACE ANALYSIS - DATA SCIENCE REPORT")
    print("=" * 60)
    print()
    
    # Load data
    data = load_results()
    total_queries = data['summary']['total_queries']
    print(f"Analyzing {total_queries} queries...\n")
    
    # Run analyses
    latency_data = analyze_latency_distribution(data)
    retrieval_data = analyze_retrieval_success(data)
    domain_counts, category_counts = analyze_domain_coverage(data)
    
    # Create visualizations
    latency_plot = create_latency_plot(latency_data)
    retrieval_plot = create_retrieval_plot(retrieval_data)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Generated visualizations:")
    print(f"  - {latency_plot}")
    print(f"  - {retrieval_plot}")
    
if __name__ == "__main__":
    main()
