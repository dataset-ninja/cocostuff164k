from typing import Dict, List, Optional, Union

from dataset_tools.templates import (
    AnnotationType,
    Category,
    CVTask,
    Domain,
    Industry,
    License,
    Research,
)

##################################
# * Before uploading to instance #
##################################
PROJECT_NAME: str = "COCO-Stuff 164k"
PROJECT_NAME_FULL: str = "COCO-Stuff 164k"
HIDE_DATASET = False  # set False when 100% sure about repo quality

##################################
# * After uploading to instance ##
##################################
LICENSE: License = License.Custom(url="https://github.com/nightrome/cocostuff#licensing")
APPLICATIONS: List[Union[Industry, Domain, Research]] = [Domain.General()]
CATEGORY: Category = Category.General(benchmark=True)

CV_TASKS: List[CVTask] = [
    CVTask.InstanceSegmentation(),
    CVTask.SemanticSegmentation(),
    CVTask.ObjectDetection(),
]
ANNOTATION_TYPES: List[AnnotationType] = [
    AnnotationType.InstanceSegmentation(),
    AnnotationType.ObjectDetection(),
]

RELEASE_DATE: Optional[str] = "2018-03-15"  # e.g. "YYYY-MM-DD"
if RELEASE_DATE is None:
    RELEASE_YEAR: int = None

HOMEPAGE_URL: str = "https://github.com/nightrome/cocostuff"
# e.g. "https://some.com/dataset/homepage"

PREVIEW_IMAGE_ID: int = 3839146
# This should be filled AFTER uploading images to instance, just ID of any image.

GITHUB_URL: str = "https://github.com/dataset-ninja/cocostuff164k"
# URL to GitHub repo on dataset ninja (e.g. "https://github.com/dataset-ninja/some-dataset")

##################################
### * Optional after uploading ###
##################################
DOWNLOAD_ORIGINAL_URL: Optional[Union[str, dict]] = {
    "COCO 2017 train images (118K images) [18GB]": "http://images.cocodataset.org/zips/train2017.zip",
    "COCO 2017 val images (5K images) [1GB]": "http://images.cocodataset.org/zips/val2017.zip",
    "Stuff+thing PNG-style annotations on COCO 2017 trainval [669MB]": "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip",
    "Stuff-only COCO-style annotations on COCO 2017 trainval [554MB]": "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuff_trainval2017.zip",
    "Thing-only COCO-style annotations on COCO 2017 trainval [241MB]": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "Indices, names, previews and descriptions of the classes in COCO-Stuff": "https://github.com/nightrome/cocostuff/blob/master/labels.md",
    "Machine readable version of the label list": "https://github.com/nightrome/cocostuff/blob/master/labels.txt",
}
# Optional link for downloading original dataset (e.g. "https://some.com/dataset/download")

CLASS2COLOR: Optional[Dict[str, List[str]]] = None
# If specific colors for classes are needed, fill this dict (e.g. {"class1": [255, 0, 0], "class2": [0, 255, 0]})

# If you have more than the one paper, put the most relatable link as the first element of the list
PAPER: Optional[Union[str, List[str]]] = [
    "https://arxiv.org/abs/1612.03716",
    "https://arxiv.org/abs/1405.0312",
]
BLOGPOST: Optional[Union[str, List[str]]] = None

CITATION_URL: Optional[str] = "https://arxiv.org/abs/1612.03716"
AUTHORS: Optional[List[str]] = ["Holger Caesar", "Jasper Uijlings", "Vittorio Ferrari"]
AUTHORS_CONTACTS: Optional[List[str]] = ["https://sites.google.com/it-caesar.de/homepage/", "h.caesar@tudelft.nl", "http://homepages.inf.ed.ac.uk/juijling", "http://calvin.inf.ed.ac.uk/members/vittoferrari"]

ORGANIZATION_NAME: Optional[Union[str, List[str]]] = [
    "University of Edinburgh, UK",
    "Google AI Perception",
]
ORGANIZATION_URL: Optional[Union[str, List[str]]] = [
    "https://www.ed.ac.uk/",
    "https://research.google/teams/perception/",
]


# Set '__PRETEXT__' or '__POSTTEXT__' as a key with value:str to add custom text. e.g. SLYTAGSPLIT = {'__POSTTEXT__':'some text}
SLYTAGSPLIT: Optional[Dict[str, Union[List[str], str]]] = {
    "__PRETEXT__": "Every image has a textual description in ***caption*** tag. Additionally, a hierarchy of the objects is contained within the ***category*** tag. Explore it in supervisely labeling tool"
}
TAGS: Optional[List[str]] = None


SECTION_EXPLORE_CUSTOM_DATASETS: Optional[List[str]] = None

##################################
###### ? Checks. Do not edit #####
##################################


def check_names():
    fields_before_upload = [PROJECT_NAME]  # PROJECT_NAME_FULL
    if any([field is None for field in fields_before_upload]):
        raise ValueError("Please fill all fields in settings.py before uploading to instance.")


def get_settings():
    if RELEASE_DATE is not None:
        global RELEASE_YEAR
        RELEASE_YEAR = int(RELEASE_DATE.split("-")[0])

    settings = {
        "project_name": PROJECT_NAME,
        "project_name_full": PROJECT_NAME_FULL or PROJECT_NAME,
        "hide_dataset": HIDE_DATASET,
        "license": LICENSE,
        "applications": APPLICATIONS,
        "category": CATEGORY,
        "cv_tasks": CV_TASKS,
        "annotation_types": ANNOTATION_TYPES,
        "release_year": RELEASE_YEAR,
        "homepage_url": HOMEPAGE_URL,
        "preview_image_id": PREVIEW_IMAGE_ID,
        "github_url": GITHUB_URL,
    }

    if any([field is None for field in settings.values()]):
        raise ValueError("Please fill all fields in settings.py after uploading to instance.")

    settings["release_date"] = RELEASE_DATE
    settings["download_original_url"] = DOWNLOAD_ORIGINAL_URL
    settings["class2color"] = CLASS2COLOR
    settings["paper"] = PAPER
    settings["blog"] = BLOGPOST
    settings["citation_url"] = CITATION_URL
    settings["authors"] = AUTHORS
    settings["authors_contacts"] = AUTHORS_CONTACTS
    settings["organization_name"] = ORGANIZATION_NAME
    settings["organization_url"] = ORGANIZATION_URL
    settings["slytagsplit"] = SLYTAGSPLIT
    settings["tags"] = TAGS

    settings["explore_datasets"] = SECTION_EXPLORE_CUSTOM_DATASETS

    return settings
