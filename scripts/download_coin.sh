mkdir -p data/coin_features
cd data/coin_features
echo "Downloading ..."
wget https://vision.eecs.yorku.ca/WebShare/COIN_s3d.zip
echo "Unzipping ..."
unzip -q COIN_s3d.zip && rm COIN_s3d.zip
# Rename COIN features to fit dataloader
echo "Renaming features ..."
python ../../tools/rename_coin_feat.py
echo "Finished!"
