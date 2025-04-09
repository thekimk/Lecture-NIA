## ✔️ Business Analytics

> Colab을 활용한 한국지능정보사회진흥원 K-ICT 빅데이터센터 비즈니스 애널리틱스 강의자료 공간입니다.

- **강사소개:** 김경원 교수 (<a href="https://sites.google.com/view/thekimk" target="_blank"><img src="https://img.shields.io/badge/Homepage-4285F4?style=flat-square&logo=Google&logoColor=white"/></a> <a href="https://scholar.google.com/citations?hl=ko&user=nHPe-4UAAAAJ&view_op=list_works&sortby=pubdate" target="_blank"><img src="https://img.shields.io/badge/Google Scholar-4285F4?style=flat-square&logo=Google Scholar&logoColor=white"/></a> <a href="https://www.youtube.com/channel/UCEYxJNI5dhnn_CdC9BEWTuA" target="_blank"><img src="https://img.shields.io/badge/YouTube-FF0000?style=flat-square&logo=YouTube&logoColor=white"/></a> <a href="https://github.com/thekimk" target="_blank"><img src="https://img.shields.io/badge/Github-181717?style=flat-square&logo=Github&logoColor=white"/></a>)

- **공지사항:**
  - 모든 강의자료는 `Github`로 공유될 예정입니다.
  - 강의자료의 `오탈자를 포함한 내용이슈`가 있어서 언제든 문의주시면 `최대한 빨리 업데이트` 할 계획입니다.
  - 강의의 전반적인 이론/실습 질문이나, 추가적 강의나 프로젝트 협업 문의는 [카카오톡채널](http://pf.kakao.com/_Exfqqb), [thekimk.kr@gmail.com](mailto:thekimk.kr@gmail.com), [032-835-8525](tel:+82328358525)로 문의해 주시면 빠르게 안내를 도와드리겠습니다.

---

## 📚 Introduction of Lecture

- **강의링크**: [[2025년 4월] 데이터 • AI 활용 역량강화과정](https://kbig.kr/portal/kbig/educationalPracticeContent/edu_seminar?bltnNo=11742800453295)

<!--
![Image](https://github.com/user-attachments/assets/2419004c-a58b-45af-a589-50ad9e6f9841)
-->

- **비즈니스 애널리틱스1:** 누구에게 마케팅을 해야할 것인가? 설명가능한 인공지능을 활용한 추천과 근거

  (1) **현재 타겟 마케팅 전략:** 누구에게 마케팅을 하여 기부를 유도할 것인가?

    > - 미래에는 이러한 특성을 가진 고객들을 타겟으로 하면 됨 (Test Explanation)
    > <p float="left">
    >   <img src="https://github.com/user-attachments/assets/807b9ec8-85f0-4e91-9b2b-8441cd24e5cd" width="600" style="margin-right: 10px;" />
    > </p>
    >
    > | **Positive Effect** |  | **Negative Effect** |  |
|:---:|:---:|:---:|:---:|
| **Train (Past)** | **Test (Future)** | **Train (Past)** | **Test (Future)** |
| 향후기부의사여부 | 향후기부의사여부 | 분류코드_가구소득1코드 | 분류코드_가구소득1코드 |
| 단체참여_종교단체여부 | 단체참여_종교단체여부 | 레저시설_관광명소유적지국립공원이용횟수 | 레저시설_관광명소유적지국립공원이용횟수 |
| 독서여부 | 독서여부 | 국내관광여행_당일여행횟수 | 국내관광여행_당일여행횟수 |
| 자원봉사활동여부 | 자원봉사활동여부 | 분류코드_가구원수 | 분류코드_가구원수 |
| 노후방법준비_본인노후준비여부 | 노후방법준비_본인노후준비여부 | 생활여건변화_전반적생활여건코드 | 생활여건변화_전반적생활여건코드 |
| - | 주관적소득수준코드 | 고용안정성코드 | 고용안정성코드 |
| 분류코드_가구소득1코드 | 분류코드_가구소득1코드 | 분류코드_세대구분코드 | 분류코드_세대구분코드 |
| 교육정도코드 | 교육정도코드 | 자식세대_계층이동코드 | 자식세대_계층이동코드 |
| 혼인상태코드 | 혼인상태코드 | 신문_인터넷구독여부 | 신문_인터넷구독여부 |
| 노후사회적관심사코드 | 노후사회적관심사코드 | 생활여견변화_사회보장제도코드 | - |
| 국내관광여행_숙박여행횟수 | 국내관광여행_숙박여행횟수 |  |  |
| 장애인복지사업견해코드 | 장애인복지사업견해코드 |  |  |
| 분류코드_연령4코드 | 분류코드_연령4코드 |  |  |
| 가구소득코드 | 가구소득코드 |  |  |
| 여가활용만족도코드 | 여가활용만족도코드 |  |  |
| 생활여건변화_보건의료서비스코드 | 생활여건변화_보건의료서비스코드 |  |  |
| - | 단체참여_시민사회단체여부 |  |  |
| 사회장애인차별정도코드 | 사회장애인차별정도코드 |  |  |
| 장애인관련시설견해코드 | 장애인관련시설견해코드 |  |  |
| - | 부채변화코드 |  |  |
| 분류코드_교육정도코드 | 분류코드_교육정도코드 |  |  |
| 소득만족도코드 | 소득만족도코드 |  |  |
| 여성취업시기코드_1.0 | 여성취업시기코드_1.0 |  |  |
| - | 주관적만족감코드 |  |  |

  (2) **미래 비즈니스 성과:** 데이터를 근거로 타겟 마케팅시 실제 기부를 할 확률은 얼마인가?

  (3) **미래 잠재고객 확보전략:** 어떤 고객들을 유입하게 할 것인가?

    > - 고객들의 실시간 마케팅 성공확률을 확인하며 타겟 후보군으로 설정 (Individual Test Explanation)
    > <p float="left">
    >   <img src="https://github.com/user-attachments/assets/81def3ea-cae4-4527-a057-194ffd27a27c" width="400" style="margin-right: 10px;" />
    >   <img src="https://github.com/user-attachments/assets/5d9ba8a3-55ac-46a9-a1f6-2dd379e1f391" width="400" />
    > </p>


---
