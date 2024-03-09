mkdir -p data/niv_features
cd data/niv_features
echo "Downloading ..."
wget https://vision.eecs.yorku.ca/WebShare/NIV_s3d.zip
echo "Unzipping ..."
unzip -q NIV_s3d.zip && rm NIV_s3d.zip
echo "Finished!"
