#!/bin/bash
# 自动重跑流程：先快速测试 → 有提升就全跑
cd /Users/taishuo/kaggle/playground_s6e6

echo "============================================"
echo "Step 0: Quick test — S2 approach with u_z+g_z"
echo "============================================"
/opt/anaconda3/bin/python3 quick_test_new_features.py

# quick_test 输出最后一行: "DELTA=+0.xxxx" 或 "DELTA=-0.xxxx"
RESULT=$(tail -1 quick_test_result.txt 2>/dev/null || echo "DELTA=0")
echo "Quick test result: $RESULT"

# 提取数字
DELTA=$(echo $RESULT | grep -oE '[-+][0-9]+\.[0-9]+' | head -1)
echo "Delta: $DELTA"

# 判断
if [ -z "$DELTA" ]; then
    echo "Error: couldn't parse delta, skipping full rerun"
    exit 1
fi

# 比较: delta > 0.0005 就全跑
if (( $(echo "$DELTA > 0.0005" | bc -l) )); then
    echo ""
    echo "============================================"
    echo "✅ +$DELTA improvement detected! Full rerun..."
    echo "============================================"
    /opt/anaconda3/bin/python3 run_all.py
    /opt/anaconda3/bin/python3 binary_chain.py
    echo "Done! All scripts rerun."
else
    echo ""
    echo "============================================"
    echo "❌ Delta too small ($DELTA), skipping full rerun"
    echo "============================================"
fi
