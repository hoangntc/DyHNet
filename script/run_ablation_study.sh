# conda activate pygnn

# Node classification
# python ../DyHNet/run_ablation_study.py --name dblp_four_area >> ../log/dblp_four_area_ablation_study_80.txt 2>&1
# python ../DyHNet/infer_eval.py --name dblp_four_area >> ../log/dblp_four_area_ablation_study_80.txt 2>&1

# python ../DyHNet/run_ablation_study.py --name yelp >> ../log/yelp_ablation_study_80.txt 2>&1
# python ../DyHNet/infer_eval.py --name yelp >> ../log/yelp_ablation_study_80.txt 2>&1

# Link prediction
python ../DyHNet/run_ablation_study.py --name dblp >> ../log/dblp_ablation_study.txt 2>&1
python ../DyHNet/infer_eval_lp.py --name dblp >> ../log/dblp_ablation_study.txt 2>&1

# python ../DyHNet/run_ablation_study.py --name imdb >> ../log/imdb_ablation_study.txt 2>&1
# python ../DyHNet/infer_eval_lp.py --name imdb >> ../log/imdb_ablation_study.txt 2>&1

