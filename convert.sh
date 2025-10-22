# Get path to the latest checkpoint
CHECKPOINTS_DIR=/nas/jiangyuxin/code/cosmos-predict2/checkpoints/cosmos_predict_v2p5/video2world/2b_calvin_debug/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

# Convert DCP checkpoint to PyTorch format
python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR