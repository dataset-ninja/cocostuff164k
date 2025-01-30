Dataset **COCO-Stuff 164k** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](https://assets.supervisely.com/remote/eyJsaW5rIjogImZzOi8vYXNzZXRzLzIwNDRfQ09DTy1TdHVmZiAxNjRrL2NvY28tc3R1ZmYtMTY0ay1EYXRhc2V0TmluamEudGFyIiwgInNpZyI6ICJMeEU3RlptYWFkMUpLOTgvc1h0bDRRNE1aZVdDbkR1QjA3ZlcrblphZzJrPSJ9)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='COCO-Stuff 164k', dst_dir='~/dataset-ninja/')
```
Make sure not to overlook the [python code example](https://developer.supervisely.com/getting-started/python-sdk-tutorials/iterate-over-a-local-project) available on the Supervisely Developer Portal. It will give you a clear idea of how to effortlessly work with the downloaded dataset.

The data in original format can be downloaded here:

- [COCO 2017 train images (118K images) [18GB]](http://images.cocodataset.org/zips/train2017.zip)
- [COCO 2017 val images (5K images) [1GB]](http://images.cocodataset.org/zips/val2017.zip)
- [Stuff+thing PNG-style annotations on COCO 2017 trainval [669MB]](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip)
- [Stuff-only COCO-style annotations on COCO 2017 trainval [554MB]](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuff_trainval2017.zip)
- [Thing-only COCO-style annotations on COCO 2017 trainval [241MB]](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [Indices, names, previews and descriptions of the classes in COCO-Stuff](https://github.com/nightrome/cocostuff/blob/master/labels.md)
- [Machine readable version of the label list](https://github.com/nightrome/cocostuff/blob/master/labels.txt)
