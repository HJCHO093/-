import xml.etree.ElementTree as elemTree
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"\\Mask_RCNN\\mrcnn")
from mrcnn.utils import Dataset, extract_bboxes
from mrcnn.visualize import display_instances
dir = os.path.dirname(os.path.realpath(__file__))
print (dir)
filename = dir + "\\kangaroo\\annots\\00001.xml"
dataset_dir = dir + "\\kangaroo\\"
## 함수영역 



## 클래스영역
class KangarooDataset(Dataset):
    def extract_boxes(self, filename):
        ## 바운딩박스 정보가 들어있는 xml파일에서 박스 정보 추출.
        ## 추가로 사진 규격도 추출
        tree = elemTree.parse(filename)
        root = tree.getroot()
        boxes = []
        # 사진규격 추출
        width = int(root.find(".//size/width").text)
        height = int(root.find(".//size/height").text)
        # 바운딩박스 정보 추출
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        return boxes, width, height 
    def load_dataset(self, dataset_dir, is_train = True):
        ## 클래스 추가(라벨링 기준)
        self.add_class("dataset", 1, "kangaroo")
         # define data locations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'
        # find all images
        for filename in os.listdir(images_dir):
        # extract image id
            image_id = filename[:-4]
        # skip bad images
            if image_id in ['00090']:
                continue
            # skip all images after 150 if we are building the train set
            if is_train and int(image_id) >= 150:
                continue
            # skip all images before 150 if we are building the test/val set
            if not is_train and int(image_id) < 150:
                continue
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
            ## add_image함수 참고. ## 인스턴스.image_info를 하면 add한 내역을 다 볼수 있음.
            '''
            def add_image(self, source, image_id, path, **kwargs):
                image_info = {
                    "id": image_id,
                    "source": source,
                    "path": path,
                }
                image_info.update(kwargs)
                self.image_info.append(image_info)
            '''
    def load_mask(self, image_id):
        ## image_info에서 add된 image중 하나의 id를 가져와 로드
        info = self.image_info[image_id]
        ## 리스트안에서 하나의 요소 딕셔너리는 아래와 같은 표현
        ''' 
        {'id': '00183', 'source':  'C:\\형종\\안용상\\kangaroo\\/annots/00180.xml'}, {'id': '00181', 'source': 'dataset', 'path': 'C:\\형종\\안용상\\kanga
        'dataset', 'path': 'C:\\형종\\안용상\\kangaroo\ \\안용상\\kangaroo\\/annots/00181.xml'}, {'id': '00182', 'source': 'dataset', 'path': 'C:\\형종\\안용상\\kangaroo\\/ima\/images/00183.jpg', 'annotation': 'C:\\형종\\ \\kangaroo\\/annots/00182.xml'}, {'id': '00183', 'source': 'dataset', 'path': 'C:\\형종\\안용상\\kangaroo\\/images/00183
        안용상\\kangaroo\\/annots/00183.xml'}
        '''
        path = info["annotation"]
        ## 위에서 정의한 extract_boxes함수 호출해서 xml파싱
        boxes, w, h = self.extract_boxes(path)
        '''박스 형태 : [xmin, ymin, xmax, ymax]'''
        ## 0행렬인 마스크 초기화 배열
        masks = np.zeros([h, w, len(boxes)], dtype='uint8')
        class_ids = []
        for i in range(len(boxes)):
            box = boxes[i]
            ## x좌표의 시작과 끝 정의
            row_s , row_e = box[1], box[3]
            ## y좌표의 시작과 끝 정의
            col_s, col_e = box[0], box[2]
            ## 초기화 마스크에서 해당 부분이 감싸는 영역만 1로 마스킹하기
            '''
            여기서 중요포인트!!! 
                * 3차원 텐서로 마스크를 만들었는데 여기서 의미가
                0축은 x축이고 1축은 y축 그리고
                2축이 한 사진에 대한 바운딩박스의 개수가 되어
                한 사진 크기에 대해, 바운딩박스개수만큼 
                여러겹이 적층되어있는 구성
            '''
            masks[row_s:row_e, col_s:col_e, i] = 1
            ## 우리 프로젝트에 적용할떄는 이부분을 수정해야함
            ## 여기서는 모든 클래스가 캥거루뿐이라 아래 단한줄의 코드로 가능했지만
            ## 우린 여러개의 음악적표현 클래스가 존재하니
            ## 바운딩박스에 해당하는 클래스에 맞게 classids에 인덱스번호를 삽입하는 
            ## 조건문 코드가 필요.
            class_ids.append(self.class_names.index('kangaroo'))
        return masks, np.asarray(class_ids, dtype='int32')
    def image_reference(self, image_id):
        ## 이미지 경로를 반환
        info = self.image_info[image_id]
        return info['path']

train_set = KangarooDataset()
train_set.load_dataset(dataset_dir)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
test_set = KangarooDataset()
test_set.load_dataset(dataset_dir, is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
for i in range(10):

    image_id = i
    ## image_info에서 image_id번쨰 인덱스의 딕셔너리를 가져와서
    ## 그 path value 를 이용해 이미지를 로드해옴
    image = train_set.load_image(image_id)
    mask, class_ids = train_set.load_mask(image_id)
    for j in range(mask.shape[2]):
        plt.imshow(image)
        # plot mask
        ## 여기서 [:,:,1]에서 1의 의미는 2번쨰 바운딩박스라는 뜻.
        plt.imshow(mask[:, :, j], cmap='gray', alpha=0.5)
        if "test" not in os.listdir():
            os.mkdir("test")
        plt.savefig("./test/test%s-%s.png"%(i,j))
# # define image id
# image_id = 1
# # load the image
# image = train_set.load_image(image_id)
# # load the masks and the class ids
# mask, class_ids = train_set.load_mask(image_id)
# # extract bounding boxes from the masks
# bbox = extract_bboxes(mask)
# # display image with masks and bounding boxes
# display_instances(image, bbox, mask, class_ids, train_set.class_names)
