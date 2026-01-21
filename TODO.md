# TODO: Fix Issues in eval_gardner_single.py

## Completed Tasks
- [x] Rename test_loader to split_loader in eval script
- [x] Update prediction collection to collect img_names and merge by Image key (GardnerDataset already returns img_name)
- [x] Enhance build_valid_mask_and_ytrue to handle "0","1","2", "A","B","C", "1","2","3" mappings
- [x] Change default --img_dir to "data/blastocyst_Dataset/Images"
- [x] Print full class ratios including missing classes (0 to num_classes-1)
- [x] Save encoded y_true in preds.csv
- [x] Ensure set_seed includes cudnn deterministic (already present)
