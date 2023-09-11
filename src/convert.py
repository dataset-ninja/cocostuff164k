# https://github.com/nightrome/cocostuff#downloads

import os
import shutil
from collections import defaultdict
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from cv2 import connectedComponents
from dataset_tools.convert import unpack_if_archive
from dotenv import load_dotenv
from pycocotools import mask as mask_util
from supervisely.io.fs import (
    file_exists,
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
)
from supervisely.io.json import load_json_file

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)
        api.file.download(team_id, teamfiles_path, local_path)

        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                api.file.download(team_id, teamfiles_path, local_path)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)

            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # project_name = "The COCO-Stuff 164"
    images_path = "/mnt/d/datasetninja-raw/coco-2017/coco2017"
    anns_path = "/mnt/d/datasetninja-raw/cocostuff164k/stuff_trainval2017"
    anns_prefix = "stuff_"
    anns_ext = ".json"
    batch_size = 30

    test = load_json_file(
        "/mnt/d/datasetninja-raw/cocostuff164k/stuff_trainval2017/stuff_val2017.json"
    )

    supercategory_to_outdoor_things = {
        "sports": ("outdoor", "things"),
        "accessory": ("outdoor", "things"),
        "animal": ("outdoor", "things"),
        "outdoor": ("outdoor", "things"),
        "vehicle": ("outdoor", "things"),
        "person": ("outdoor", "things"),
    }

    supercategory_to_indoor_things = {
        "indoor": ("indoor", "things"),
        "appliance": ("indoor", "things"),
        "electronic": ("indoor", "things"),
        "furniture": ("indoor", "things"),
        "food": ("indoor", "things"),
        "kitchen": ("indoor", "things"),
    }

    supercategory_to_outdoor_stuff = {
        "water": ("outdoor", "stuff"),
        "ground": ("outdoor", "stuff"),
        "solid": ("outdoor", "stuff"),
        "sky": ("outdoor", "stuff"),
        "plant": ("outdoor", "stuff"),
        "structural": ("outdoor", "stuff"),
        "building": ("outdoor", "stuff"),
    }

    supercategory_to_indoor_stuff = {
        "food": ("indoor", "stuff"),
        "textile": ("indoor", "stuff"),
        "furniture": ("indoor", "stuff"),
        "window": ("indoor", "stuff"),
        "floor": ("indoor", "stuff"),
        "ceiling": ("indoor", "stuff"),
        "wall": ("indoor", "stuff"),
        "rawmaterial": ("indoor", "stuff"),
    }

    supercategores = [
        supercategory_to_outdoor_things,
        supercategory_to_indoor_things,
        supercategory_to_outdoor_stuff,
        supercategory_to_indoor_stuff,
    ]

    def create_ann(image_path):
        labels = []

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        ann_data = image_name_to_ann_data[get_file_name_with_ext(image_path)]

        for curr_ann_data in ann_data:
            subcategories_values = []
            category_id = curr_ann_data[0]
            obj_class = idx_to_obj_class[category_id]
            curr_class_name = obj_class.name
            if curr_class_name == "person":
                subcategories_values = ("outdoor", "things")
            elif curr_class_name == "other":
                subcategories_values = "other"

            else:
                subcategory = class_to_supercategory.get(curr_class_name)
                subcategories_values.append(subcategory)
                if subcategory in ["food", "furniture"]:
                    if curr_class_name in [
                        "food-other",  # door and desk are the same in both parent subcategories, but they are not in this dataset :)
                        "vegetable",
                        "salad",
                        "fruit",
                        "furniture-other",
                        "stairs",
                        "light",
                        "counter",
                        "mirror",
                        "cupboard",
                        "cabinet",
                        "shelf",
                        "table",
                    ]:
                        parent_subcategory = ("indoor", "stuff")
                else:
                    for supercategores_data in supercategores:
                        parent_subcategory = supercategores_data.get(subcategory)
                        if parent_subcategory is not None:
                            subcategories_values.extend(parent_subcategory)

            if type(subcategories_values) is not str:
                subcategories_values = ", ".join(subcategories_values)
            tag_category = sly.Tag(tag_category_meta, value=subcategories_values)

            polygons_coords = curr_ann_data[1]
            mask = mask_util.decode(polygons_coords)
            ret, curr_mask = connectedComponents(mask.astype("uint8"), connectivity=8)
            for i in range(1, ret):
                obj_mask = curr_mask == i
                curr_bitmap = sly.Bitmap(obj_mask)
                if curr_bitmap.area > 30:
                    curr_label = sly.Label(curr_bitmap, obj_class, tags=[tag_category])
                    labels.append(curr_label)

            bbox_coord = curr_ann_data[2]
            rectangle = sly.Rectangle(
                left=int(bbox_coord[0]),
                top=int(bbox_coord[1]),
                right=int(bbox_coord[0] + bbox_coord[2]),
                bottom=int(bbox_coord[1] + bbox_coord[3]),
            )
            label_rectangle = sly.Label(rectangle, obj_class, tags=[tag_category])
            labels.append(label_rectangle)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels)

    tag_category_meta = sly.TagMeta("category", sly.TagValueType.ANY_STRING)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(tag_metas=[tag_category_meta])

    idx_to_obj_class = {}
    class_to_supercategory = {}

    for ds_name in ["val2017", "train2017"]:
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        ds_path = os.path.join(images_path, ds_name)

        image_id_to_name = {}
        image_id_to_name_tag = {}
        image_name_to_ann_data = defaultdict(list)
        images_names = os.listdir(ds_path)
        ann_path = os.path.join(anns_path, anns_prefix + ds_name + anns_ext)

        if file_exists(ann_path):
            ann = load_json_file(ann_path)
            for curr_category in ann["categories"]:
                if idx_to_obj_class.get(curr_category["id"]) is None:
                    class_name = curr_category["name"]
                    if class_to_supercategory.get(class_name) is None:
                        parent = curr_category["supercategory"]
                        if class_name != parent:
                            class_to_supercategory[class_name] = parent
                    obj_class_poly = sly.ObjClass(class_name, sly.AnyGeometry)
                    meta = meta.add_obj_class(obj_class_poly)
                    idx_to_obj_class[curr_category["id"]] = obj_class_poly
            api.project.update_meta(project.id, meta.to_json())

            for curr_image_info in ann["images"]:
                image_id_to_name[curr_image_info["id"]] = curr_image_info["file_name"]

            for curr_ann_data in ann["annotations"]:
                image_id = curr_ann_data["image_id"]
                image_name_to_ann_data[image_id_to_name[image_id]].append(
                    [
                        curr_ann_data["category_id"],
                        curr_ann_data["segmentation"],
                        curr_ann_data["bbox"],
                    ]
                )

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for images_names_batch in sly.batched(images_names, batch_size=batch_size):
            img_pathes_batch = [
                os.path.join(ds_path, image_name) for image_name in images_names_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, images_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(image_path) for image_path in img_pathes_batch]
            api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(images_names_batch))
    return project
