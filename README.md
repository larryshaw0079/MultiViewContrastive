# Co-training Contrastive Coding for Self-supervised Sleep Stage Classification

## Quick Start

Run Moco R1d

```bash
python main_moco.py --data-path data/sleepedf_39.lmdb/ --meta-file data/sleepedf_39_split/meta.pkl --network r1d --time-len 3000 --fold 0 --kfold 10 --optimizer sgd --device 0
```

Run Moco R2d

```bash
python main_moco.py --data-name sleepedf --data-path ./data/sleepedf_39_hht.lmdb/ --meta-file ./data/sleepedf_39_hht_split/meta.pkl --network r2d --num-extend 10 --time-len 30 --freq-len 100 --fold 0 --kfold 10 --optimizer sgd --devices 0
```

Run CoCLR R1d R2d

```bash
python main_coclr.py --data-name sleepedf --data-path-v1 ./data/sleepedf_39.lmdb/ --data-path-v2 ./data/sleepedf_39_hht.lmdb/ --meta-file-v1 ./data/sleepedf_39_split/meta.pkl --meta-file-v2 ./data/sleepedf_39_hht_split/meta.pkl --load-path-v1 ./cache/moco_r1d/moco_run_0_pretrained.pth.tar --load-path-v2 ./cache/moco_r2d/moco_run_0_pretrained.pth.tar --fold 0 --kfold 10 --optimizer sgd
```

