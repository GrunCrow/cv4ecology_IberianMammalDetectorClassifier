# Computer Vision Documentation

<!--- This template is intended to help you track the important choices you make throughout your project, which are necessary for publishing a CV paper in a transparent and reproducible way.

Estimated min 4 pages (workshop paper length), preferred max 8 pages. No need to hit the max - concise wording is great!-->

## Concise description of task 
<!--Write 1 phrase on task, study area, image type
e.g., “semantic segmentation of penguin guano in the Antarctic from Sen-2 satellite imagery”, “multi-class classification of predators on Vancouver island in camera trap imagery”-->

The "Automated Census and Biodiversity Monitoring System using Deep Learning" is an ongoing project focused on developing an advanced protocol and workflow for wildlife monitoring using camera trapping, citizen science, and deep learning techniques. The primary objective of this project is to create a powerful Neural Network for species classification using camera-trap images obtained from Doñana National Park.

By leveraging cutting-edge deep learning methods, the project aims to achieve unbiased estimates of species and community dynamics. This will enable cost-effective and prompt responses to ecological changes, ultimately contributing to the conservation and understanding of biodiversity in the region.


<!-- =============================================== -->


## Motivation
<!--Why is your task/method important, why your proposal could be a valuable research direction in that field (NOTE: can pull from original proposal)-->


<!-- =============================================== -->


## Description of data

#### How was the data collected?

#### Where was the data collected?

#### Over what time period?

#### Which sensors were used? 

#### Are there any sensor configurations that need to be mentioned for full reproducibility?

<!--*E.g., satellite imagery from Sentinel 2b MSI Level-2A
How was the data labeled, and, if appropriate, what level of approximate error is there in the labels?*
(Think about how this might be estimated)-->

#### What are the dimensions of your data?

#### How many instances does your train/val/test dataset have? e.g., 10.000 train; 2.500 val; and 2.500 test images. 

#### How many channels does your data have? E.g. 4-channel imagery red, green, blue, and near-infrared

<!--e.g., 100 satellite swaths with 6000x6000 pixels and 4-channels (RGB-NIR) tiled into a total of 10.000 image chips of size 128x128. The dataset totals 31GB including all train/val/test splits.  -->

#### Visual examples of the data, across the variation of what is seen

#### Is your dataset balanced or what is the distribution per class of your dataset?

#### What domain does your dataset cover?

#### Description of data splitting (train/val/test)

#### What claims does this dataset and data split enable you to make? 

#### What are possible limitations of models trained on this data with this split?


<!-- =============================================== -->


## *Brief* Literature Review

#### What is the unique research gap that you’re addressing?

#### Think of what you’ve discussed/read in the reading groups and how it pertains to your work; how is your project similar/different?


<!-- =============================================== -->


## Architecture choices

#### One end-to-end model or a concatenation of models?

#### What are the inputs and outputs of the model? (shape, what they contain)

#### If possible via a jamboard or an ipad sketch, can you illustrate what the components of the pipeline look like?

#### Data augmentation choices + any other tricks used (optional)

#### List of augmentations explored and final choices, discussion of why these augmentations are appropriate

#### Final hyperparameters

#### Including details to replicate full hyperparameter search

#### Loss function used

#### Relevant metrics to report

#### Choice of evaluation metric(s) and intuitive description of what these will measure for your ecological task

#### More fine-grained metrics per class

#### Confusion matrix

#### PR curves


<!-- =============================================== -->


## Visualizations
<!--E.g. Segmentation masks, GradCAM
Examples of failures, discussion of limitations
 Where and how would it be valid to claim this model could be used?-->


<!-- =============================================== -->


## Resource usage 
<!--I.e. I used a single NVIDIA A5000 which took X hours to run one experiment; I ran Y experiments for a total of Z GPU hours-->

#### Inference speed+computational needs of final model end-to-end

#### Future directions

#### Where would you like to see this project go?

#### Where (if applicable), would you like to adapt what you’ve learned at the summer school to future projects
<!--For the Resnicks (optional, to be used by the funders)-->

#### How would you say the workshop furthered your research? 

#### What impact will it have going forward?


