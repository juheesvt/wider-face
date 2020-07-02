
"""  wider face evaluate를 위한 txt 파일 생성 script  """

import json
import os

prediction_dict = None
ground_truth = None

# model's prediction for validation dataset
with open("/workspace/object_detection_detectron2/output/coco_instances_results.json") as prediction_json:
    prediction_dict = json.load(prediction_json)
    
# ground truth for validation dataset
with open("/workspace/object_detection_detectron2/detectron2/data/wider_face_val_annot_coco_style.json") as ground_truth_json:
    ground_truth_dict = json.load(ground_truth_json)
    


for gt_img in ground_truth_dict["images"]:
    
    file_name = gt_img["file_name"][:-3] + "txt"
    path = "/workspace/object_detection_detectron2/prediction/images/" + file_name
    img_id = gt_img["id"]
    sub_path = os.path.dirname(path)
    
    # 해당 directory가 없으면 생성해주는 코드
    if not os.path.exists(sub_path):
        os.mkdir(sub_path)
        
    with open(path, "w") as output:
        
        img_count = 0
        bbox_output_list = []
        
        for prediction_info in prediction_dict:
            if prediction_info["image_id"] == img_id:
                img_count += 1
                prediction_bbox = list(map(str, prediction_info["bbox"]))
                prediction_score = str(prediction_info["score"])
                bbox_output_list.append(" ".join(prediction_bbox + [prediction_score]))
            
        output.write(file_name + "\n")
        output.write(str(img_count) + "\n")
        for bbox_output in bbox_output_list:
            output.write(bbox_output + "\n")
