from  dlrm_s_pytorch import *
import torch

model_path='/Data/chinmay/dlrm/criteo_pretrain_model/tb0875_10M.pt'
pretrain_model=torch.load(model_path)

dlrm = DLRM_Net(
    m_spa=64,  # Based on the embedding size seen (64)
    ln_emb=[9980333, 36084, 17217, 7378, 20134, 3, 7112, 1442, 61, 9758201, 1333352, 313829, 10, 2208, 11156, 122, 4, 970, 14, 9994222, 7267859, 9946608, 415421, 12420, 101, 36],
    ln_bot=[13, 512, 256, 64, 16],
    ln_top=[512, 256, 1],
    arch_interaction_op="dot",  # Use the setting you had in args
    arch_interaction_itself=False,  # Example setting
    sigmoid_bot=-1,
    sigmoid_top=1,
    sync_dense_params=False,
    loss_threshold=0.0,
    ndevices=1,
    qr_flag=False,
    qr_operation="mult",
    qr_collisions=0,
    qr_threshold=200,
    md_flag=False,
    md_threshold=200,
    weighted_pooling=None,
    loss_function="bce"
)
