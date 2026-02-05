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
    st.title("기존 고객 유지 전략 분석")
    tab1, tab2, tab3 = st.tabs(["전략 1: 마케팅 분야", "전략 2: 라이브 스트리밍 콘텐츠 생성", "전략 3: 번들링 및 결합 상품 확대"])

    with tab1:
        st.subheader('데이터 기반 고객 유지 전략')
        with st.expander('1. 3개월 이상 구독 유지', expanded=True):
            st.markdown('### 3개월 이상 구독 유지 고객 대상 리텐션 프로그램')

            cols1, cols2 = st.columns([3,1])
            with cols1:
                st.write('**전략 내용**')
                st.info('3개월 이상 구독을 유지한 고객에 한해 **구독 해지 시 1개월 무료 체험권 제공**')
            with cols2:
                st.metric('예상 이탈 감소','15%', delta='-15%', delta_color='inverse')
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

