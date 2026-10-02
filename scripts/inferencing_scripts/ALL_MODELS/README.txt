In this folder, you choose which models are being inferenced by save_raw_all.py

Keep in mind that the save_raw_all.py script is currently designed to overwrite
existing images.


Your model files should have the .pt and .json a 3 part naming convention, currently these are
storing "{MODEL NAME}_{EXPERIMENT TYPE}_{INITIAL INPUT}". For example "LPD_SparseView_FBP". When 
being inferenced, the class import asscociated with the keyword "LPD" in the save_all_raw.py or 
general_stats.py file will be loaded.
