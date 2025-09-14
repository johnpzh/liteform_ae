export FLUSH_L2=ON

python proc05.spmm.end-to-end.py \
-s models/model.for_selection.train_set.RandomForest.joblib \
-p models/model.for_partitions.train_set.RandomForest.joblib