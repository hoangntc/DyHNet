conda activate pygnn
# python ../DyHNet/tune_params.py --name dblp >> ../log/dblp.txt 2>&1
# python ../DyHNet/train.py --name dblp_four_area >> ../log/dblp_four_area.txt 2>&1
# python ../DyHNet/train.py --name yelp >> ../log/yelp_train.txt 2>&1
# python ../DyHNet/train.py --name dblp_lp >> ../log/dblp_lp.txt 2>&1
python ../DyHNet/train.py --name yelp_lp # >> ../log/yelp_lp.txt 2>&1
