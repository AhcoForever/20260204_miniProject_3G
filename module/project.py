import streamlit as st
import pandas as pd
import numpy as np
# 민영 수정
# ======================================================== 1.페이지 설정 =============================================================

st.set_page_config(
    page_title="넷플릭스 대시보드",
    layout="wide",  
    initial_sidebar_state="expanded"
)

# ======================================================== 2. 데이터 =================================================================
@st.cache_data 
def load_data():
    data = {
        'Month': pd.date_range(start='2025-01-01', periods=12, freq='MS'),
        'Subscribers': [1250, 1270, 1321, 1345, 1360, 1393, 1410, 1425, 1440, 1465, 1490, 1516],
        'Retention': [12.1, 12.5, 11.8, 13.0, 14.2, 13.5, 12.8, 13.2, 14.5, 15.0, 14.8, 15.2], # 유지기간 데이터
        'Churn_Rate': [2.1, 2.3, 3.5, 2.0, 1.8, 2.5, 2.1, 1.9, 1.7, 2.2, 2.0, 1.5]             # 이탈률 데이터
    }
    df = pd.DataFrame(data)

    # 성장률 계산
    df['Prev_Subscribers'] = df['Subscribers'].shift(1)
    df['Growth_Rate'] = ((df['Subscribers'] - df['Prev_Subscribers']) / df['Prev_Subscribers']) * 100
    return df

in_df = load_data()


# ======================================================== 3.사이드바 구성=============================================================
with st.sidebar:
    st.header("🔍 분석 설정")

    month_labels = [d.strftime('%Y년 %m월') for d in in_df['Month']]

    selected_month = st.selectbox("분석 월 선택", options=month_labels, index=0)
    selected_plan = st.selectbox("요금제 필터", ['광고형', '스탠다드', '프리미엄'], index=0)
    analysis = st.button("🚀 데이터 분석 실행", use_container_width=True)
    st.divider()
    st.info(f"💡[현재 설정]   기간: **{selected_month}**,  요금제: **{selected_plan}**")

# ======================================================== 4. 메인화면 구성=============================================================

header_col1, header_col2 = st.columns([1.5, 6])
col1, col2, col3 = st.columns(3)
# 페이지 상태 초기화
if 'page' not in st.session_state:
    st.session_state.page='home'
# 페이지 전환 함수 
def go_to_page(page_name):
    st.session_state.page = page_name

# 메인 화면
if st.session_state.page == 'home':
    with header_col1:
        st.image('https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg', width=250)

    with header_col2:
        st.title('넷플릭스 구독자 현황 분석')
        st.text('💡 데이터로 추적하는 넷플릭스 구독자들의 이탈 신호와 유지 전략')

    with col1 : 
        if st.button('구독자 이탈 현상 분석'):
            go_to_page('subscription_analysis')

    with col2 : 
        if st.button('구독자 이탈 원인 진단'):
            go_to_page('reason')       

    with col3 : 
        if st.button('고객 유지 전략'):
            go_to_page('retention')

# 구독자 분석 탭
elif st.session_state.page == 'subscription_analysis' :
    # 뒤로가기 버튼
    if st.button("홈으로 돌아가기"):
        go_to_page('home') 

    with header_col1:
        st.image('https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg', width=250)

    with header_col2:
        st.title('구독자 이탈 현상 분석')
        st.text('계정 사용 기간별 분석')


# 원인 진단 탭
elif st.session_state.page == 'reason':
    # 뒤로가기 버튼
    if st.button("홈으로 돌아가기"):
        go_to_page('home') 

    with header_col1:
        st.image('https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg', width=250)

    with header_col2:
        st.title('구독자 이탈 원인 진단')
        st.text('이탈률이 가장 높은 조합과 낮은 조합을 파악하여 타겟 마케팅에 활용')


# 기존 고객 유지 전략 페이지
elif st.session_state.page =='retention':

    # 뒤로가기 버튼
    if st.button("홈으로 돌아가기"):
        go_to_page('home')

    with header_col1:
        st.image('https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg', width=250)

    with header_col2:
        st.title('기존 고객 유지 전략')
        st.text('기존 고객 유지를 위한 전략 및 이탈 방지 시뮬레이션')

    tab1, tab2, tab3 = st.tabs(["전략 1: 마케팅 분야", "전략 2: 서비스 모델의 변화", "전략 3: 유통 및 플랫폼 전략"])

    # 전략 1: 3개월 구도 유지 시 혜택
    with tab1:
        st.subheader('데이터 기반 고객 유지 전략')
        with st.expander('1. 3개월 이상 구독 유지 혜택 제공', expanded=True):
            st.markdown('### 3개월 이상 구독 유지 고객 대상 리텐션 프로그램')

            cols1, cols2 = st.columns([3,1])
            with cols1:
                st.write('**전략 내용**')
                st.info('3개월 이상 구독을 유지한 고객에 한해 **구독 해지 시 1개월 무료 체험권 제공**')
            with cols2:
                st.metric('예상 이탈 감소','15%', delta='-15%', delta_color='inverse')
            st.write("")
            st.write("**기대 효과:**")
            st.markdown(
                """
                - 해지 시점에 인센티브 제공으로 재가입 유도
                - 브랜드 충성도 강화
                """
            )
            st.write('**실행 방안:**')
            st.markdown(
                """
                1. 해지 버튼 클릭 시 팝업으로 "1개월 무료 혜택" 제안
                2. 해지 완료 후 재가입 유도 이메일 발송
                3. 3개월 구독 유지 시 자동으로 혜택 안내
                """
            )
            st.image("data/1month_benefit.png", width = 400)

    # 전략 2: 라이브 스트리밍
    with tab2 : 
        st.subheader('VOD에서 라이브 스트리밍으로의 확장')
        with st.expander('2. 라이브 스트리밍 컨텐츠 추가', expanded=True):
            st.markdown('### 스포츠 생중계 및 독점 라이브러리 강화')
            col1, col2 = st.columns([3,1])
            with col1 : 
                st.write('**전략 내용:**')
                st.info("드라마나 영화와 달리 **'휘발성'이 강하고 '본방사수'가 필요한 스포츠 컨텐츠로 고정 시청층 확보")

            with col2:
                st.metric('락인 효과', '높음', delta='팬덤 기반')
            st.write("")
            st.write('**실제 사례:**')

            case_col1, case_col2 = st.columns(2)
            with case_col1:
                st.markdown("**쿠팡 플레이**")
                st.write('- 프리미어리그 독점 중계')
                st.write('- 축구 팬 고정 확보')
                st.write('- 시즌 중 해지율 극소')

            with case_col2:
                st.markdown('**티빙**')
                st.write('- KBO 야구 중계')
                st.write('- 테니스 독점 콘텐츠')
                st.write('- 스포츠 팬층 타겟팅')
            st.success('**핵심 인사이트**: 특정 시즌 동안은 해지할 수 없는 강력한 팬덤 기반의 락인 구현')

            st.write('**추천 콘텐츠**')
            st.markdown("""
                        - ⚽️ 글로벌 축구 리그 (EPL, 라리가 등)
                        - ⚾️ 국내외 야구 중계 (KBO, MLB)
                        - 🏀 농구 (NBA, KBL)
                        - 🎮  e스포츠 대회 생중계
                        """)
    # 전략 3: 번들링 및 결합 상품 확대
    with tab3 : 
        st.subheader('번들링 및 결합 상품 확대')
        with st.expander('3. 번들링 및 결합 상품 확대', expanded=True):
            st.markdown('### 타 서비스와의 전략적 제휴')
            col1, col2 = st.columns([3,1])
            with col1 : 
                st.write('**전략 내용:**')
                st.info('단독 구독의 부담을 낮추기 위해 타 서비스와 혜택을 묶는 방식')
            with col2 : 
                st.metric('해지 장벽', '상승', delta='일상 밀착')
            st.write('')
            st.write('**실제 사례:**')
            st.image("data/tving.png", width = 400)
            st.markdown("""
                        - **티빙 X 배달의 민족** (배민클럽)
                            - OTT + 배달 할인 결합
                            - 일상 생활 밀착형 서비스
                        """)
            st.image('data/wave.jpg', width = 400)
            st.markdown(
                """
                 - **티빙 X 웨이브** 합병 수준의 결합 상품
                    - 콘텐츠 라이브러리 확대
                    - 구독료 부담 확산
                """
            )
            st.success("**핵심 인사이트**: 라이프스타일 인프라와를 통한 락인(Lock-in)극대화 - 서비스 이탈 시 체감되는 유틸리티 손실 강조")
            st.write('**추천 제휴 파트너:**')           
            partner_col1, partner_col2, partner_col3 = st.columns(3)
            with partner_col1:
                st.markdown("**🥘 배달/외식**")
                st.write('- 요기요')
                st.write('- 스타벅스')
            with partner_col2 : 
                st.markdown('*🚗 모빌리티**')
                st.write('- 카카오T')
                st.write('- 타다')
                st.write('- 쏘카')
            with partner_col3:
                st.markdown('**📱 통신/유틸리티**')
                st.write('- SKT/KT/LG')
                st.write('- 네이버 플러스')
                st.write('- 쿠팡 로켓와우')

st.divider()

# ======================================================== 5. 분석 로직 =================================================================

if analysis:
    month_num = int(selected_month.split(' ')[1].replace('월', ''))
    target_df = in_df.iloc[month_num - 1:month_num] 

    if not target_df.empty:
        latest_data = target_df.iloc[0]
        latest_mau = latest_data['Subscribers'] / 100  
        growth_rate = latest_data['Growth_Rate']

        st.subheader(f"📊 {selected_month} 분석 결과 (요금제: {selected_plan})")

        col3, col4, col5 = st.columns(3)
        with col3:
            delta_text = f"{growth_rate:.2f}% (전월 대비)" if pd.notnull(growth_rate) else "신규 데이터"
            st.metric(
                label="📈 월 가입자 수", 
                value=f"{latest_mau:.2f} M", 
                delta=delta_text
            )
            st.caption("(가입자 수): 전체 체급 지표")

        with col4:
            st.metric(label="⏳ 유지 기간", value=f"{latest_data['Retention']}개월", delta="0.5개월")
            st.caption("(유지 기간): 수익성 지표")

        with col5:
            st.metric(label="🚨 이탈률", value=f"{latest_data['Churn_Rate']}%", delta="-0.3%", delta_color="inverse")
            st.caption("(이탈률): 위기 신호 지표")

        st.divider()

