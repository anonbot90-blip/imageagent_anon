#!/bin/bash

# Compare Standard vs RL Training Data
# Shows differences in data generation and resulting datasets

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Paths
STANDARD_DATA="$PROJECT_ROOT/training_data/standard/planner_training_data.json"
RL_DATA="$PROJECT_ROOT/training_data/rl/planner_training_data.json"

print_color $PURPLE "═══════════════════════════════════════════════════════════════════════════════"
print_color $PURPLE "📊 COMPARISON: Standard vs RL Training Data"
print_color $PURPLE "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# PART 1: Script Configuration Comparison
# ============================================================================

print_color $CYAN "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
print_color $CYAN "1️⃣  SCRIPT CONFIGURATION COMPARISON"
print_color $CYAN "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cat << 'EOF'
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STANDARD DATA GENERATION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ Script: generate_standard_training_data.sh                                  │
│                                                                              │
│ Configuration:                                                               │
│   • NUM_DATAPOINTS = 1555                                                    │
│   • RL_MODE = false                                                          │
│   • RL_THRESHOLD = 3.5 (IGNORED - RL_MODE is false)                         │
│   • RL_METRIC = "overall_quality" (IGNORED - RL_MODE is false)              │
│   • OUTPUT_DIR = training_data/standard/                                     │
│                                                                              │
│ Python Command:                                                              │
│   python scripts/generate_planner_training_data.py \                        │
│       imageagent_results_5000 \                                              │
│       --output-dir training_data/standard \                                  │
│       --num-datapoints 1555                                                  │
│                                                                              │
│ Behavior:                                                                    │
│   ✅ Takes first 1555 samples (no quality filtering)                        │
│   ✅ Includes ALL samples regardless of reward score                        │
│   ✅ Does NOT read reward_scores.json                                       │
│   ✅ Mixed quality: includes both good and bad samples                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                       RL DATA GENERATION                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Script: generate_rl_training_data.sh                                        │
│                                                                              │
│ Configuration:                                                               │
│   • NUM_DATAPOINTS = 3000                                                    │
│   • RL_MODE = ALWAYS TRUE (hardcoded via --rl-data flag)                    │
│   • THRESHOLD = 4.0 (DEFAULT)                                                │
│   • METRIC = "overall_quality"                                               │
│   • OUTPUT_DIR = training_data/rl/                                           │
│                                                                              │
│ Python Command:                                                              │
│   python scripts/generate_planner_training_data.py \                        │
│       imageagent_results_5000 \                                              │
│       --output-dir training_data/rl \                                        │
│       --num-datapoints 3000 \                                                │
│       --rl-data \                                                            │
│       --threshold 4.0 \                                                      │
│       --reward-metric overall_quality                                        │
│                                                                              │
│ Behavior:                                                                    │
│   ✅ FILTERS samples by reward score >= 4.0                                 │
│   ✅ Only includes HIGH-QUALITY samples                                     │
│   ✅ REQUIRES reward_scores.json files                                      │
│   ✅ Stops after finding 3000 qualifying samples (or exhausts dataset)      │
└─────────────────────────────────────────────────────────────────────────────┘
EOF

echo ""

# ============================================================================
# PART 2: Key Differences
# ============================================================================

print_color $CYAN "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
print_color $CYAN "2️⃣  KEY DIFFERENCES"
print_color $CYAN "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cat << 'EOF'
┌───────────────────────────┬─────────────────────────┬─────────────────────────┐
│         Feature           │   Standard Data         │      RL Data            │
├───────────────────────────┼─────────────────────────┼─────────────────────────┤
│ Quality Filtering         │ ❌ NO                   │ ✅ YES                  │
│ Threshold                 │ N/A                     │ >= 4.0 (default)        │
│ Requires reward_scores    │ ❌ NO                   │ ✅ YES                  │
│ Max samples requested     │ 1555                    │ 3000                    │
│ Actual samples (typical)  │ ~1555 (all taken)       │ ~1555 (after filtering) │
│ Quality distribution      │ Mixed (1.0-5.0)         │ High only (4.0-5.0)     │
│ RL_MODE flag              │ false (ignored)         │ true (via --rl-data)    │
│ Output directory          │ training_data/standard/ │ training_data/rl/       │
│ Use case                  │ Standard models         │ RL models               │
└───────────────────────────┴─────────────────────────┴─────────────────────────┘
EOF

echo ""

# ============================================================================
# PART 3: Actual Data Comparison
# ============================================================================

print_color $CYAN "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
print_color $CYAN "3️⃣  ACTUAL DATA FILES COMPARISON"
print_color $CYAN "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if files exist
if [ ! -f "$STANDARD_DATA" ]; then
    print_color $RED "❌ Standard data not found: $STANDARD_DATA"
    STANDARD_EXISTS=false
else
    STANDARD_EXISTS=true
    print_color $GREEN "✓ Standard data found: $STANDARD_DATA"
fi

if [ ! -f "$RL_DATA" ]; then
    print_color $RED "❌ RL data not found: $RL_DATA"
    RL_EXISTS=false
else
    RL_EXISTS=true
    print_color $GREEN "✓ RL data found: $RL_DATA"
fi

echo ""

if [ "$STANDARD_EXISTS" = true ] && [ "$RL_EXISTS" = true ]; then
    
    # Count samples
    if command -v jq >/dev/null 2>&1; then
        STANDARD_COUNT=$(jq '.total_samples' "$STANDARD_DATA")
        RL_COUNT=$(jq '.total_samples' "$RL_DATA")
        
        STANDARD_FILTERED=$(jq '.rl_filtered' "$STANDARD_DATA")
        RL_FILTERED=$(jq '.rl_filtered' "$RL_DATA")
        
        print_color $YELLOW "📊 Sample Counts:"
        echo "   Standard: $STANDARD_COUNT samples (RL filtered: $STANDARD_FILTERED)"
        echo "   RL:       $RL_COUNT samples (RL filtered: $RL_FILTERED)"
        echo ""
        
        # File sizes
        STANDARD_SIZE=$(du -h "$STANDARD_DATA" | cut -f1)
        RL_SIZE=$(du -h "$RL_DATA" | cut -f1)
        
        print_color $YELLOW "💾 File Sizes:"
        echo "   Standard: $STANDARD_SIZE"
        echo "   RL:       $RL_SIZE"
        echo ""
        
        # Sample one entry from each
        print_color $YELLOW "🔍 Sample Entry Comparison:"
        echo ""
        
        print_color $BLUE "Standard Data (First Entry):"
        jq '.samples[0] | {id, user_prompt, num_actions: (.target_action_plan.actions | length)}' "$STANDARD_DATA"
        echo ""
        
        print_color $BLUE "RL Data (First Entry):"
        jq '.samples[0] | {id, user_prompt, num_actions: (.target_action_plan.actions | length)}' "$RL_DATA"
        echo ""
        
        # Check for overlap
        print_color $YELLOW "🔗 Sample Overlap Analysis:"
        
        # Extract image IDs from both datasets
        STANDARD_IDS=$(jq -r '.samples[].id' "$STANDARD_DATA" | sort)
        RL_IDS=$(jq -r '.samples[].id' "$RL_DATA" | sort)
        
        # Count overlapping IDs
        OVERLAP_COUNT=$(comm -12 <(echo "$STANDARD_IDS") <(echo "$RL_IDS") | wc -l)
        STANDARD_ONLY=$(comm -23 <(echo "$STANDARD_IDS") <(echo "$RL_IDS") | wc -l)
        RL_ONLY=$(comm -13 <(echo "$STANDARD_IDS") <(echo "$RL_IDS") | wc -l)
        
        echo "   Samples in BOTH datasets:    $OVERLAP_COUNT"
        echo "   Samples ONLY in Standard:    $STANDARD_ONLY"
        echo "   Samples ONLY in RL:          $RL_ONLY"
        echo ""
        
        # Calculate percentages
        if [ "$STANDARD_COUNT" -gt 0 ]; then
            OVERLAP_PCT=$(echo "scale=1; $OVERLAP_COUNT * 100 / $STANDARD_COUNT" | bc)
            print_color $CYAN "   → ${OVERLAP_PCT}% of Standard samples are in RL dataset"
        fi
        
        if [ "$RL_COUNT" -gt 0 ]; then
            RL_OVERLAP_PCT=$(echo "scale=1; $OVERLAP_COUNT * 100 / $RL_COUNT" | bc)
            print_color $CYAN "   → ${RL_OVERLAP_PCT}% of RL samples are in Standard dataset"
        fi
        
    else
        print_color $YELLOW "⚠️  Install 'jq' for detailed comparison: sudo apt install jq"
    fi
fi

echo ""

# ============================================================================
# PART 4: Summary & Implications
# ============================================================================

print_color $CYAN "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
print_color $CYAN "4️⃣  SUMMARY & IMPLICATIONS"
print_color $CYAN "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cat << 'EOF'
🎯 WHY THE DIFFERENCE MATTERS:

1. Data Quality:
   • Standard: Trains on ALL samples (including low-quality ones)
   • RL: Trains ONLY on high-quality samples (reward >= 4.0)
   
2. Model Performance:
   • Standard model learns from mixed-quality examples
   • RL model learns from curated high-quality examples
   • RL model should perform better due to cleaner training signal

3. Current Issue (Nov 11, 2025):
   ⚠️  PROBLEM: generate_standard_training_data.sh has:
       NUM_DATAPOINTS="1555"  (limits to 1555 samples)
       RL_THRESHOLD=3.5       (but RL_MODE=false, so ignored)
   
   ⚠️  PROBLEM: This means Standard data is ALSO limited to 1555 samples,
       but WITHOUT quality filtering. It's just taking the first 1555
       samples regardless of quality.

4. Expected Behavior:
   ✅ Standard: Should use ALL available samples (~2800)
   ✅ RL: Should filter to high-quality samples (~1555 after filtering)
   
5. Recommendation:
   • Change generate_standard_training_data.sh:
     NUM_DATAPOINTS=""  (empty = use all samples)
   • Keep generate_rl_training_data.sh as is:
     THRESHOLD=4.0 (only high-quality samples)

EOF

print_color $PURPLE "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# PART 5: Deep Analysis with Python
# ============================================================================

print_color $CYAN "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
print_color $CYAN "5️⃣  DEEP ANALYSIS (Python)"
print_color $CYAN "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$STANDARD_EXISTS" = true ] && [ "$RL_EXISTS" = true ]; then
    COMPARISON_OUTPUT="$PROJECT_ROOT/training_data/comparison"
    
    print_color $BLUE "Running detailed analysis..."
    echo ""
    
    python "$PROJECT_ROOT/scripts/training/analyze_training_data_diff.py" \
        "$STANDARD_DATA" \
        "$RL_DATA" \
        "$PROJECT_ROOT/imageagent_results_5000" \
        "$COMPARISON_OUTPUT"
    
    PYTHON_EXIT=$?
    
    if [ $PYTHON_EXIT -eq 0 ]; then
        echo ""
        print_color $GREEN "✅ Deep analysis complete!"
        echo ""
        print_color $YELLOW "📁 Results saved to: $COMPARISON_OUTPUT/"
        echo ""
        print_color $CYAN "Generated files:"
        echo "   • overlap_samples.json"
        echo "   • standard_only_samples.json"
        echo "   • rl_only_samples.json"
        echo "   • quality_distribution.json"
        echo "   • quality_distribution.png"
        echo "   • action_complexity_stats.json"
        echo "   • COMPARISON_REPORT.md"
        echo ""
        print_color $PURPLE "📖 Read the full report:"
        print_color $PURPLE "   cat $COMPARISON_OUTPUT/COMPARISON_REPORT.md"
    else
        print_color $RED "❌ Python analysis failed (exit code: $PYTHON_EXIT)"
    fi
else
    print_color $YELLOW "⚠️  Skipping deep analysis (missing data files)"
fi

echo ""
print_color $PURPLE "═══════════════════════════════════════════════════════════════════════════════"
print_color $GREEN "✅ Comparison Complete"
print_color $PURPLE "═══════════════════════════════════════════════════════════════════════════════"
echo ""

