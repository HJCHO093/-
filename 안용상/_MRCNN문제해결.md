### mrcnn 모듈 다운
1. git clone https://github.com/matterport/Mask_RCNN.git
2. cd Mask_RCNN
python setup.py install
3. 설치확인
> pip show mask-rcnn
### mrcnn 파이썬 라이브러리 경로에 추가
> 1. 시스템 환경변수 편집
>> * 새로만들기 버튼 클릭
>> * PYTHONPATH 이름으로 만들고
>> * 경로는 "C:\Users\qhfkd\anaconda3\Lib\site-packages\mrcnn"
이런 형태로 삽입( 즉 mrcnn이 들어있는 폴더 삽입)
<img src ="./others/syspath.png" width = 500/>
### mrcnn 모듈 불러오기
> from mrcnn.utils import Dataset
해보기


## 필요한 라이브러리 버전(버전 동기화를 위해)
텐서플로우 1.15.3권장 ( pip버전이 낮아야댐 .)
케라스 2.2.4 권장
파이썬 3.7 미만 권장 ( 3.6.9 )
>>파이썬 버전 낮추고 pip 버전 낮춰서 텐서플로우 1.15.3까는법https://velog.io/@kijh30123/Tensorflow-version-1.15.5in-colab-%EC%BD%94%EB%9E%A9%EC%97%90%EC%84%9C-%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C-%EB%B2%84%EC%A0%84-%EB%82%AE%EC%B6%94%EA%B8%B0
>>파이썬 3.6.8설치 https://www.zinnunkebi.com/python-download-install-exec/#google_vignette
>>보다 낮은 버전으로 환경변수 수정 방법 https://foon.tistory.com/31

>>파이썬 및 pip 버전 낮춘뒤에 
>>> pip uninstall tensorflow
>>> pip uninstall keras
>>> pip install tensorflow==1.15.3
>>> pip install keras==2.2.4
>>>>다시깔아야함.