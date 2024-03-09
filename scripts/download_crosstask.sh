mkdir -p data/crosstask_features
cd data/crosstask_features
echo "Downloading ..."
wget https://vision.eecs.yorku.ca/WebShare/CrossTask_s3d.zip
echo "Unzipping ..."
unzip -q CrossTask_s3d.zip && rm CrossTask_s3d.zip
echo "Finished!"
