▼ params
fold_dir "./ml-10M100K/folds_hash"
device "cuda"
batch_size 131072
seed 42
shuffle_chunk 1000000
rank mf 64
epochs mf 30
Ir_mf 0.02
reg_mf 0.02
reg_bias 0.005
rank_pgd 32
iters_pgd 30
eta_pgd 0.2
rank_cvx 32
iters_softimpute 30
lam_softimpute 0.5
iters_ista 30
lam_ista 0.5
eta_ista 0.1
iters_fista 30
lam fista 0.5
eta_fista 0.1
iters_fw30
tau_fw 50