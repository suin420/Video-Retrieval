# 🎞️ DeepScreen: Al Movie Rating System
<br><br>
## 배경 및 목적
현재 영화의 상영 등급은 주로 수작업으로 결정되며, 이 과정에서 많은 시간과 비용이 소모되고, 사람의 주관적인 판단에 의존하게 된다. 이러한 방식은 효율성과 객관성의 한계가 있다.
이러한 문제점을 해결하기 위해 본 프로젝트에서는 AI 기반의 영화 상영 등급 자동화 시스템을 개발했다. 본 시스템은 Video Retrieval 기술을 활용하여 영화 속 폭력성, 선정성, 약물 복용 등의 요소를 분석하고, 이를 바탕으로 상영 등급을 자동으로 산정하는 AI 프로세스를 구현했다.
<br><br>
## 주최/주관 & 팀원
- 인공지능 학회 X:AI 5기 Adv Session Multi-Modal 2조
- 학회원 5명
<br><br>
## 프로젝트 기간
2024.07 ~ 2024.08 (2개월)
<br><br>
## 프로젝트 소개
<img width="1000" src="https://github.com/user-attachments/assets/70db2230-04e5-48e8-addb-2196df80781a">

1. 입력: 영화 데이터(비디오)

2. 비디오 분할: OpenCV를 사용하여 비디오를 10초 단위로 분할 (30분 분량의 영상은 약 1분 정도 소요)
   
4. Video Retrieval: 생성된 클립들에 대해 주어진 텍스트와 관련된 클립을 검색하고, 유사도가 일정 threshold를 넘는 클립을 선택하여 폭력성, 선정성 등 평가 기준에 해당하는 영상 클립을 추출
   
6. 상영 등급 산출: 각 평가 기준별로 추출된 클립들의 지속 시간을 바탕으로 최종 상영 등급과 기준을 계산하여 결과 산출

<br>

<img width="700" src="https://github.com/user-attachments/assets/89516172-d660-4eb6-b601-e3dcbbd3a36d">

Video Retrieval을 수행할 모델로 Text is Mass을 선정하였다. <br>
Text is Mass 모델은 텍스트와 비디오의 피처를 추출하여 유사도를 측정하여 텍스트 쿼리와 가장 의미 있는 비디오 클립을 찾을 수 있다는 장점을 가진다.
<br><br>

<img width="700" src="https://github.com/user-attachments/assets/08ad037a-8232-4ecf-b261-32faa8a54c39">

finetuning에 사용한 데이터셋은 세 가지이다. 데이터 정제 과정을 거쳐 finetuning에 활용한 최종적인 학습 데이터는 2,773개이다. 
<br><br>

<img width="700" src="https://github.com/user-attachments/assets/2ec37525-7144-4a6c-a4dc-869b51ca0c15">

finetuning 결과이다. <br>
테스트 데이터셋은 영화 데이터셋인 할리우드만을 활용하여 한 번, 그리고 전체 데이터 셋에 대해 양을 조절해가며, 총 4개의 버전으로 구성하였다. <br>
전체 데이터셋 이외에도 hollywoods 데이터셋 만으로 확인한 이유는, 최종적으로 영화 데이터셋에 대해 인퍼런스를 진행하기 때문에, 영화 데이터셋에 대한 성능을 비교해보기 위함이었다. 
<br><br>

<img width="700" src="https://github.com/user-attachments/assets/92091c76-bba7-4aaa-8f08-f5aa47b295c7">

action keywords extraction 단계에서는 객관적인 등급 산정을 위해 RAG 파이프라인을 구축하였다. 사용된 Document는 온라인 등급 분류 서비스의 등급분류 HTML이다. <br>
주제, 선정성, 폭력성, 대사, 공포, 약물, 모방위험 (7개)의 영화 등급 산정 기준 중, 선정성, 폭력성, 약물이 등급 분류에서 주요한 기준으로 채택하였으며, 그 외의 모방 위험이나 공포 요소는 다소 객관적이지 않으므로 제외하였다. <br>
또한 위 논문을 바탕으로 모방 위험과 관련성이 높은, 담배와 알코올을 위험 행동의 새로운 기준으로 추가하였다.
<br><br>

## 역할
- Video Retrieval 선행 연구 조사 및 모델 탐색
- CLIP4CLIP 모델 분석 및 인퍼런스
- 파인튜닝 데이터셋 탐색 및 전처리
- Gradio 툴을 이용한 인터페이스 구축 및 프로젝트 발표
<br><br>
## 발표 자료
- [중간발표자료](https://github.com/user-attachments/files/17958612/ADV-Toyproject-.pdf)
- [최종발표자료](https://github.com/user-attachments/files/17958623/MM2_._.pdf)
- [발표 영상](https://youtu.be/zUbHDxC-Mi8?si=CwR3Zko64dhNUk8A)
