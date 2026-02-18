#!/bin/bash

#######################################################
# Quick Test Script for SC-ELAN YAML Files
#######################################################

echo "=================================================="
echo "Testing SC-ELAN YAML Configuration Files"
echo "=================================================="
echo ""

# Array of SC-ELAN variants
variants=(
    "yolo11-scelan"
    "yolo11-scelan-dilated"
    "yolo11-scelan-slim"
    "yolo11-scelan-hybrid"
)

# Test each variant
for variant in "${variants[@]}"; do
    echo "Testing: ${variant}.yaml"
    
    # Check if file exists
    if [ ! -f "${variant}.yaml" ]; then
        echo "  ❌ ERROR: File not found!"
        continue
    fi
    
    # Try to parse the model (dry run)
    python -c "
from ultralytics.nn.tasks import parse_model, yaml_model_load
import torch

try:
    # Load YAML
    cfg = yaml_model_load('${variant}.yaml')
    print('  ✓ YAML loaded successfully')
    
    # Parse model
    model, savelist = parse_model(cfg, ch=3, verbose=False)
    print('  ✓ Model parsed successfully')
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'  ✓ Total parameters: {total_params:,}')
    
    # Test forward pass
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        y = model(x)
    print(f'  ✓ Forward pass successful: {tuple(x.shape)} -> {tuple(y.shape)}')
    
    # Count SC-ELAN modules
    sc_elan_count = sum(1 for m in model if 'SC_ELAN' in str(type(m)))
    print(f'  ✓ SC-ELAN modules found: {sc_elan_count}')
    
except Exception as e:
    print(f'  ❌ ERROR: {e}')
    import traceback
    traceback.print_exc()
" 2>&1
    
    echo ""
done

echo "=================================================="
echo "Testing completed!"
echo "=================================================="
