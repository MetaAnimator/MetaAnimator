from torch import nn
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
import insightface
from insightface.app import FaceAnalysis
import numpy as np
from PIL import Image
import torch
import cv2
import os
from tqdm import tqdm


# 判断两个embedding的相似度，用余弦相似度（Cosine Similarity）
def cosine_similarity(embedding1, embedding2):
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

class FaceModel(nn.Module):
    def __init__(self):
        super(FaceModel, self).__init__()
        self.app = FaceAnalysis(
            name='antelopev2', root='/root/autodl-tmp/projects/MetAine/scripts/', providers=['CUDAExecutionProvider', 'CPUExecutionProvider', ]
        )
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        self.handler_ante = insightface.model_zoo.get_model('/root/autodl-tmp/projects/MetAine/scripts/models/antelopev2/glintr100.onnx')
        self.handler_ante.prepare(ctx_id=-1)

        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device="cpu",
        )
        self.face_helper.face_parse = init_parsing_model(model_name='bisenet', device="cpu")

def get_face_embedding(face_model, input_image_path):
    no_face_flag = False
    face_model.face_helper.clean_all()
    input_img = cv2.imread(input_image_path)
    input_img_bgr = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    face_info = face_model.app.get(input_img_bgr)
    rimg = face_model.app.draw_on(input_img, face_info)
    # cv2.imwrite("./t1_output.jpg", rimg)

    if len(face_info) > 0: 
        # 选择最大的人脸
        face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]  # select largest face (if more than one detected)
        face_embedding = face_info['embedding']
        # print("\n\n提取到了人脸embedding：", face_embedding.shape)
    else:
        face_embedding = None
        no_face_flag = True

    if face_embedding is None:
        face_model.face_helper.read_image(input_img_bgr)
        face_model.face_helper.get_face_landmarks_5(only_center_face=True)
        face_model.face_helper.align_warp_face()

        if len(face_model.face_helper.cropped_faces) == 0:
            face_embedding = np.zeros((512,))
        else:
            validation_image_align_face = face_model.face_helper.cropped_faces[0]
            print('fail to detect face using insightface, extract embedding on align face')
            face_embedding = face_model.handler_ante.get_feat(validation_image_align_face)
    return face_embedding, no_face_flag

def cal_face_sim(face_model, reference_image_path, gen_img_dir):
    print("Calculating face similarity:", reference_image_path)
    reference_embedding, _ = get_face_embedding(face_model, reference_image_path)
    sim_list = []
    no_face_num = 0
    for item in tqdm(os.listdir(gen_img_dir)):
        gen_img_path = os.path.join(gen_img_dir, item)
        gen_embedding, no_face_flag = get_face_embedding(face_model, gen_img_path)
        if no_face_flag:
            no_face_num += 1
            continue
        similarity = cosine_similarity(reference_embedding, gen_embedding)
        # print(f"Image: {gen_img_path}, Cosine Similarity: {similarity}")
        sim_list.append(similarity)
    print(f"Average Cosine Similarity: {np.mean(sim_list)}")
    print(f"No face num: {no_face_num}", "Total image num: ", len(os.listdir(gen_img_dir)))
    return np.mean(sim_list)

    