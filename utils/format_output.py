import numpy as np
import json
from typing import List

from pydantic import BaseModel

class ImagePred(BaseModel):
    classes: List[str]
    prob: List[float]
    top_2: dict

# Format model prediction to JSON
# for easy-to-read api response.
def format_image_pred(pred, classes):
    output = {}
    output['classes'] = classes
    output['prob'] = pred[0].tolist()
    output['top_2'] = {}
    top_2_ind = pred[0].argsort()[::-1][:2]
    top_2_prob = pred[0][top_2_ind]
    top_2_classes = np.array(classes)[top_2_ind]
    for index, key in enumerate(top_2_classes):
        # output["top_2"].append({key: top_2_prob[index].item()})
        output["top_2"][key] = top_2_prob[index].item()
    output = ImagePred(classes=output['classes'], prob=output['prob'], top_2=output["top_2"])
    print(output)
    return output