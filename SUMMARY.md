**COCO-Stuff 164k** is a dataset for instance segmentation, semantic segmentation, and object detection tasks. It is applicable or relevant across various domains. 

The dataset consists of 163957 images with 4691398 labeled objects belonging to 172 different classes including *other*, *person*, *tree*, and other: *sky-other*, *wall-concrete*, *clothes*, *building-other*, *metal*, *grass*, *wall-other*, *pavement*, *furniture-other*, *table*, *road*, *window-other*, *textile-other*, *chair*, *car*, *dining table*, *light*, *plastic*, *fence*, *ceiling-other*, *dirt*, *bush*, *clouds*, *paper*, *plant-other*, and 144 more.

Images in the COCO-Stuff 164k dataset have pixel-level instance segmentation and bounding box annotations. Due to the nature of the instance segmentation task, it can be automatically transformed into a semantic segmentation task (only one mask for every class). There are 40677 (25% of the total) unlabeled images (i.e. without annotations). There are 3 splits in the dataset: *test2017* (40670 images), *train2017* (118287 images), and *val2017* (5000 images). Additionally, an hierarchy of the objects is contained within the ***category*** tag. Explore it in supervisely. The dataset was released in 2018 by the University of Edinburgh, UK and Google AI Perception.

<img src="https://github.com/dataset-ninja/cocostuff164k/raw/main/visualizations/poster.png">
