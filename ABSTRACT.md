The authors of the **COCO-Stuff 164k** dataset discuss the significance of semantic classes, which can be categorized as either **thing** classes (objects with well-defined shapes, e.g., car, person) or **stuff** classes (amorphous background regions, e.g., grass, sky). They note that while much attention has been given to thing classes in classification and detection works, stuff classes have received less focus. However, they emphasize that stuff classes play a crucial role in understanding images, providing information about scene type, the likely presence and location of thing classes through contextual reasoning, physical attributes, material types, and geometric properties of the scene.

The COCO-Stuff 164k dataset supplements the [COCO 2017]("https://arxiv.org/abs/1405.0312") dataset with pixel-wise annotations for 91 stuff classes. It contains 172 classes in total: 80 thing, 91 stuff, and 1 class _unlabeled_. The 80 thing classes are the same as in COCO 2017. The 91 stuff classes are curated by an expert annotator. The class "unlabeled" is used in two situations: 1) if a label does not belong to any of the 171 predefined classes, or 2) if the annotator cannot infer the label of a pixel.

The hierarchy of labels:

<img src="https://github.com/supervisely/supervisely/assets/78355358/d3c78712-cd5b-496b-91ff-7fb0beafceb7" alt="image" width="600">

Authors argue that stuff classes are essential as they constitute the majority of the visual environment, determine scene types, influence the understanding of thing classes' locations, and contribute to depth ordering and relative positions of things.

Furthermore, they detail the protocol used for stuff labeling, emphasizing the efficiency of superpixel-based annotation and its accuracy compared to polygon-based annotation. They analyze the impact of boundary complexity on annotation time and highlight that superpixels offer a substantial improvement in annotation efficiency while maintaining accuracy.

In conclusion, the authors stress the importance of stuff classes in scene understanding, showcasing dataset's value in augmenting the understanding of stuff-thing interactions in complex images. They also provide insights into the efficiency and accuracy of their annotation protocol.
