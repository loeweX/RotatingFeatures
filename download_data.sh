#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
cd datasets

# Download 4Shapes datasets.
wget https://zenodo.org/record/8324835/files/RotatingFeatures_4Shapes.zip
unzip RotatingFeatures_4Shapes.zip
rm RotatingFeatures_4Shapes.zip


# Download FoodSeg103 datasets.
wget https://research.larc.smu.edu.sg/downloads/datarepo/FoodSeg103.zip
unzip -P LARCdataset9947 FoodSeg103.zip
rm FoodSeg103.zip


# Download Pascal VOC 2012 dataset and add trainaug split.
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip

tar -xf VOCtrainval_11-May-2012.tar
unzip SegmentationClassAug.zip -d VOCdevkit/VOC2012

mv trainaug.txt VOCdevkit/VOC2012/ImageSets/Segmentation
mv VOCdevkit/VOC2012/SegmentationClassAug/* VOCdevkit/VOC2012/SegmentationClass/

rm -r VOCdevkit/VOC2012/__MACOSX
rm SegmentationClassAug.zip
rm VOCtrainval_11-May-2012.tar

cd ..