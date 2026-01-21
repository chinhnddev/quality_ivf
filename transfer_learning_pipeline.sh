#!/bin/bash
# Example: Complete Transfer Learning Pipeline
# Stage Pretraining → Fine-tuning on EXP/ICM/TE tasks

set -e

OUTPUT_BASE="outputs/transfer_learning_demo"
SEED=42

echo "=========================================="
echo "Stage 1: Pretraining (Development Stages)"
echo "=========================================="

python scripts/train_stage_pretrain.py \
    --epochs 40 \
    --output_dir "${OUTPUT_BASE}/stage_pretrain" \
    --seed ${SEED} \
    --batch_size 32 \
    --lr 3e-4

PRETRAIN_CKPT="${OUTPUT_BASE}/stage_pretrain/best.ckpt"
echo "✓ Pretraining complete. Checkpoint: ${PRETRAIN_CKPT}"

echo ""
echo "=========================================="
echo "Stage 2: Fine-tune EXP (Developmental Competence)"
echo "=========================================="

python scripts/train_gardner_single.py \
    --task exp \
    --config configs/gardner/base.yaml \
    --track improved \
    --use_coral 1 \
    --use_weighted_sampler 1 \
    --pretrain_ckpt "${PRETRAIN_CKPT}" \
    --out_dir "${OUTPUT_BASE}/exp_transfer" \
    --seed ${SEED}

echo "✓ EXP fine-tuning complete. Output: ${OUTPUT_BASE}/exp_transfer"

echo ""
echo "=========================================="
echo "Stage 3: Fine-tune ICM (Inner Cell Mass)"
echo "=========================================="

python scripts/train_gardner_single.py \
    --task icm \
    --config configs/gardner/base.yaml \
    --track improved \
    --use_weighted_sampler 1 \
    --pretrain_ckpt "${PRETRAIN_CKPT}" \
    --out_dir "${OUTPUT_BASE}/icm_transfer" \
    --seed ${SEED}

echo "✓ ICM fine-tuning complete. Output: ${OUTPUT_BASE}/icm_transfer"

echo ""
echo "=========================================="
echo "Stage 4: Fine-tune TE (Trophectoderm)"
echo "=========================================="

python scripts/train_gardner_single.py \
    --task te \
    --config configs/gardner/base.yaml \
    --track improved \
    --use_weighted_sampler 1 \
    --pretrain_ckpt "${PRETRAIN_CKPT}" \
    --out_dir "${OUTPUT_BASE}/te_transfer" \
    --seed ${SEED}

echo "✓ TE fine-tuning complete. Output: ${OUTPUT_BASE}/te_transfer"

echo ""
echo "=========================================="
echo "All stages complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Pretraining: ${OUTPUT_BASE}/stage_pretrain/best.ckpt"
echo "  - EXP (with CORAL + sampler): ${OUTPUT_BASE}/exp_transfer"
echo "  - ICM (with sampler): ${OUTPUT_BASE}/icm_transfer"
echo "  - TE (with sampler): ${OUTPUT_BASE}/te_transfer"
