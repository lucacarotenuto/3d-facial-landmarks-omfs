# 3D Facial landmark detection for Oral and maxillofacial surgery
MSc project by Luca Carotenuto

### Data Preparation
1. Execute scripts/prep_pcl.py to convert .obj meshes to point clouds and optionally sample for lower resolution
2. Execute scripts/ldmks.py to summarize all landmarks in single pickle file ldmks.pkl 
3. Execute scripts/ptwise_targets.py to create point-wise targets  for training


### Initial network
4. Run diffusion-net/experiments/headspace_ldmks/headspace_ldmks.py for training the model and for inference

### Prepare Refined Data
5. Run scripts/create_refined_train.py to create refined training samples based on the landmarks
6. Run scripts/create_refined_test.py to create refined testing samples based on prediction from initial network

### Refined network
6. Run diffusion-net/experiments/refine_ldmks/refine_ldmks.py for training the model and do inference

### Visualize or print results
7. Run scripts/colorize_vertices.py to create visualizations in .txt format viewable in meshlab or with pptk
8. Run scripts/compute_accuracy.py to print individual errors, sample/landmark mean and total mean errors
