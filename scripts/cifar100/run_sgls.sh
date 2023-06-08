DATASET=cifar100
DATA_ROOT='~/dataset/cifar100'
ARCH=resnet18
LR=0.1
LR_SCHEDULE='step'
MILESTONES='100 150'
EPOCHS=200
BATCH_SIZE=128
LOSS=sgls
ALPHA=0.9
ES=90
NUM_STEPS=10
EXP_NAME=${DATASET}/adv_${ARCH}_${LOSS}_m${ALPHA}_p${ES}_$1
SAVE_DIR=ckpts/${EXP_NAME}
LOG_FILE=logs/${EXP_NAME}.log
GPU_ID='0'

### print info
echo ${EXP_NAME}
mkdir -p ckpts
mkdir -p logs

### train
CUDA_VISIBLE_DEVICES=${GPU_ID} \
python -u main_adv.py --arch ${ARCH} --loss ${LOSS} \
        --num-steps ${NUM_STEPS} \
        --sat-es ${ES} --sat-alpha ${ALPHA} \
        --dataset ${DATASET} --data-root ${DATA_ROOT} \
        --lr ${LR} --lr-schedule ${LR_SCHEDULE} --lr-milestones ${MILESTONES} \
        --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} \
        --save-dir ${SAVE_DIR} \
        >> ${LOG_FILE} 2>&1

