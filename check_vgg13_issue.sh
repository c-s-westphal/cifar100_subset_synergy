#!/bin/bash
# Quick check script for VGG13 job issue

echo "=== Checking VGG13 job script ==="
echo "GPU request line:"
grep "gpu=" scripts/job_manager_vgg13.sh
echo ""
echo "Device argument:"
grep "device" scripts/job_manager_vgg13.sh | tail -1
echo ""
echo "=== All job scripts GPU requests ==="
grep -H "gpu=" scripts/job_manager_vgg*.sh
