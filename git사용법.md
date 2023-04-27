# 우리 프로젝트에 깃 사용법

## 원격 레포지토리 clone하기
git clone https://github.com/HJCHO093/-.git

## 원격 레포지토리의 새 변경사항 반영해서 내 로컬 저장소에(본인 컴퓨터) 반영하기
git pull
> * 만약 오류나면 git add . ; git commit -m "." ; 명령어가 필요할 수 있다

## 내 로컬저장소에서 작업후 원격 레포지토리에 변경사항 업로드하기
1. git add .
2. git commit -m "변경사항에 대한 코멘트"
3. git push origin main

## 깃에 대해 잘 설명해주는 사이트
> * https://git-scm.com/book/ko/v2/%EC%8B%9C%EC%9E%91%ED%95%98%EA%B8%B0-Git-%EA%B8%B0%EC%B4%88
>> 이곳에서 깃의 기본 컨셉, 기본명령어 (init, add ,commit, config, push, pull, clone)들을 살펴보면 좋다.
