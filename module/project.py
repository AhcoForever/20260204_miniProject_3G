import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import platform
import seaborn as sns
from lifelines import KaplanMeierFitter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter
# 민영 수정
# ======================================================== 1.페이지 설정 =============================================================

st.set_page_config(
    page_title="넷플릭스 대시보드",
    layout="wide",  
    initial_sidebar_state="expanded"
)
# 맥 환경 폰트 깨짐 방지
plt.rcParams['font.family'] = 'AppleGothic'
# 윈도우 환경 폰트 깨짐 방지
#font_path = "C:/Windows/Fonts/malgun.ttf"
# font_name = fm.FontProperties(fname=font_path).get_name()
# plt.rc('font', family=font_name)
# plt.rcParams['axes.unicode_minus'] = False
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

# Page1: 구독자 분석 탭
elif st.session_state.page == 'subscription_analysis' :
    # 뒤로가기 버튼
    if st.button("홈으로 돌아가기"):
        go_to_page('home') 

    with header_col1:
        st.image('https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg', width=250)

    with header_col2:
        st.title('📉 이탈 예측 모델링 및 골등타임 도출')
        st.text("데이터가 말해주는 '언제', '누구를', '어떻게' 잡아야 하는가")
        st.markdown('---')

    # def set_korean_font():
    #     system = platform.system()

    #     if system == 'Darwin':  # macOS
    #         plt.rc('font', family='AppleGothic')
    #     elif system == 'Windows':  # Windows
    #         plt.rc('font', family='Malgun Gothic')
    #     else:  # Linux
    #         plt.rc('font', family='NanumGothic')

    #     plt.rc('axes', unicode_minus=False)

    # set_korean_font()

    # =================================================================
    # 📊 1. [막대+선] 이탈 4주 전 행동 변화 (골든타임)
    # =================================================================
    st.header("1. 이탈 골든타임 ")
    st.info("💡 이탈 확정 유저들의 4주간 행동 패턴 추적 결과")

    weeks = ['4주 전', '3주 전', '2주 전', '1주 전']
    frequency = [5.2, 4.1, 2.3, 0.8]  # 접속 횟수 (막대)
    completion = [75, 60, 45, 20]     # 완독률 (선)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig1, ax1 = plt.subplots(figsize=(10, 6))

        # 1) 막대 그래프 (접속 횟수)
        bars = ax1.bar(weeks, frequency, color='#000000', label='주간 접속 횟수', alpha=0.7, width=0.5)
        ax1.set_ylabel("주간 접속 횟수 (회)", fontsize=12)
        ax1.set_ylim(0, 6)

        # 2) 선 그래프 (완독률) - 축 공유 (twinx)
        ax2 = ax1.twinx()
        line = ax2.plot(weeks, completion, color='#E50914', marker='o', linewidth=3, markersize=10, label='콘텐츠 완독률')
        ax2.set_ylabel("완독률 (%)", fontsize=12, color='#E50914')
        ax2.tick_params(axis='y', labelcolor='#E50914')
        ax2.set_ylim(0, 100)

        # 3) 'Warning' 마크 표시 (2주 전 시점)
        # 2주 전은 index 2
        ax2.annotate('Warning\n(Golden Time)', 
                    xy=(2, 45), xytext=(2, 65),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    ha='center', fontsize=12, fontweight='bold', color='red')

        # 범례 합치기
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')

        ax1.set_title("이탈 D-4주 행동 변화 추이", fontsize=15)
        st.pyplot(fig1)

    with col2:
        st.markdown("""
        **[데이터 포인트]**
        * **4주 전**: 접속 5.2회, 완독률 75% (정상)
        * **2주 전**: 접속 2.3회, 완독률 45% (**급감**)
        * **결론**: 접속 횟수가 반토막 나고, 완독률이 50% 밑으로 떨어지는 **'2주 전'**이 마케팅이 개입해야 할 유일한 골든타임입니다.
        """)

    st.markdown("---")

    # =================================================================
    # 📉 2. [산점도] 이탈 위험군 식별 (Retention vs Recency)
    # =================================================================
    st.header("2. 위험군 식별: \"14일의 법칙 (Red-line)\"")
    st.info("💡 마지막 접속일(Recency) 경과에 따른 이탈 확률 상관관계")

    # 산점도 데이터 생성 (트렌드를 보여주기 위한 가상 데이터 생성)
    np.random.seed(42)
    recency_days = np.random.randint(1, 31, 200) # 1~30일 경과한 유저 200명
    # 이탈 확률 함수 (S커브 형태: 7일에 45%, 14일에 82%가 되도록 조정)
    def churn_prob(day):
        # 로지스틱 함수 변형
        base_prob = 1 / (1 + np.exp(-(day - 8) * 0.4)) 
        # 약간의 노이즈 추가 (산점도처럼 보이게)
        noise = np.random.normal(0, 0.05)
        prob = base_prob + noise
        return np.clip(prob * 100, 0, 100)

    churn_probs = [churn_prob(d) for d in recency_days]
    df_scatter = pd.DataFrame({'Recency': recency_days, 'ChurnProb': churn_probs})

    col3, col4 = st.columns([2, 1])

    with col3:
        fig2, ax3 = plt.subplots(figsize=(10, 6))

        # 산점도 그리기
        # 14일 기준 색상 구분 (Red Line 넘으면 빨강)
        colors = ['red' if x >= 14 else 'blue' for x in df_scatter['Recency']]
        ax3.scatter(df_scatter['Recency'], df_scatter['ChurnProb'], c=colors, alpha=0.6, edgecolors='w', s=80)

        # 레드라인 (x=14)
        ax3.axvline(x=14, color='red', linestyle='--', linewidth=2)
        ax3.text(14.5, 10, '이탈 레드라인\n(14일)', color='red', fontsize=12, fontweight='bold')

        # 주요 포인트 텍스트 (7일, 14일)
        # 실제 데이터 포인트 근사치에 표시
        ax3.annotate('7일 경과\n(이탈확률 45%)', xy=(7, 45), xytext=(2, 60),
                    arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)
        ax3.annotate('14일 경과\n(이탈확률 82%)', xy=(14, 82), xytext=(16, 90),
                    arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10, fontweight='bold', color='red')

        ax3.set_title("마지막 접속 경과일(Recency) vs 이탈 확률", fontsize=15)
        ax3.set_xlabel("마지막 접속 후 경과일 (Day)")
        ax3.set_ylabel("이탈 확률 (%)")
        ax3.set_xlim(0, 31)
        ax3.set_ylim(0, 105)
        ax3.grid(True, linestyle='--', alpha=0.5)

        st.pyplot(fig2)

    with col4:
        st.markdown("""
        **[Red-Line 분석]**
        * **7일 차**: 이탈 확률 45% (주의 단계)
        * **14일 차**: 이탈 확률 **82%** (복구 불가능)
        * **전략**: 사용자가 **7일~14일 사이** 구간에 진입했을 때, 강력한 푸시 알림과 복귀 혜택을 쏴야 합니다. 14일이 지나면 돌아오지 않습니다.
        """)

    st.markdown("---")

    # =================================================================
    # 🍕 3. [파이 차트] 현재 구독자 상태 분포
    # =================================================================
    st.header("3. 현재 구독자 진단: \"우리는 누구에게 집중해야 하는가\"")
    st.info("💡 행동 데이터를 기반으로 분류한 전체 구독자 현황")

    # 데이터 설정
    labels = ['안정군 (Active)', '주의군 (At-risk)', '위험군 (Churn-imminent)']
    sizes = [70, 20, 10]
    colors = ['#4CAF50', '#FF9800', '#F44336'] # 초록, 주황, 빨강
    explode = (0, 0, 0.1)  # 위험군(10%)만 툭 튀어나오게 강조

    col5, col6 = st.columns([1, 1])

    with col5:
        fig3, ax4 = plt.subplots(figsize=(8, 8))

        wedges, texts, autotexts = ax4.pie(sizes, explode=explode, labels=labels, colors=colors,
                                        autopct='%1.1f%%', shadow=True, startangle=140,
                                        textprops={'fontsize': 12})

        # 텍스트 스타일 꾸미기
        plt.setp(autotexts, size=14, weight="bold", color="white")

        ax4.set_title("전체 구독자 리스크 등급 분포", fontsize=15)
        st.pyplot(fig3)

    with col6:
        st.markdown("#### 📋 그룹별 정의 및 Action Plan")
        st.success("**🟢 안정군 (Active) - 70%**\n* 주 3회 이상 접속, 완독률 70% 이상\n* **Action**: 건드리지 않음 (Natural Retention)")
        st.warning("**🟠 주의군 (At-risk) - 20%**\n* 접속 주기 불규칙, 검색만 하고 시청 안 함\n* **Action**: '찜한 콘텐츠' 알림, 인기작 추천")
        st.error("**🔴 위험군 (Churn-imminent) - 10%**\n* **7일 이상 미접속**, 3개월 차 진입\n* **Action**: **즉시 개입!** (특별 할인 쿠폰, 1:1 메시지)")



# Page2 : 원인 진단
elif st.session_state.page == 'reason':
    st.set_page_config(layout="wide")

    # 뒤로가기 버튼
    if st.button("홈으로 돌아가기"):
        go_to_page('home') 

    with header_col1:
        st.image('https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg', width=250)

    with header_col2:
        st.title('구독자 이탈 원인 진단')
        st.text('이탈률이 가장 높은 조합과 낮은 조합을 파악하여 타겟 마케팅에 활용')
        st.subheader('OTT Churn Analytics Dashboard')

    # 데이터 로드
    df = pd.read_csv("data/Subscription_Service_Churn_Dataset.csv")
    df['tenure'] = df['AccountAge']
    df['churn'] = df['Churn']
    df['long_term'] = df['tenure'] >= 6

    st.header("3개월 이탈 구조")

    kmf = KaplanMeierFitter()
    kmf.fit(df['tenure'], event_observed=df['churn'])

    fig1, ax1 = plt.subplots(figsize=(7,5)) 
    kmf.plot_survival_function(ax=ax1, linewidth=3)
    ax1.axvline(3, color='red', linestyle='--')
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)
    ''

    # 사용자 시청 패턴
    st.header("사용자 시청 패턴")

    fig2, ax2 = plt.subplots(figsize=(7,5))
    sns.scatterplot(
        data=df,
        x='ViewingHoursPerWeek',
        y='tenure',
        hue='long_term',
        alpha=0.6,
        ax=ax2
    )
    ax2.axvline(10, linestyle='--')
    ax2.axhline(6, linestyle='--')
    ax2.set_title("Magic Moment: Viewing vs Survival")
    st.pyplot(fig2)

    df['engagement_score'] = (
        df['ViewingHoursPerWeek'] * 0.4 +
        df['ContentDownloadsPerMonth'] * 0.3 +
        df['WatchlistSize'] * 0.2 +
        df['UserRating'] * 0.1
    )

    threshold = 30
    magic_users = df[df['ViewingHoursPerWeek'] >= threshold]

    baseline = df['long_term'].mean()
    magic_prob = magic_users['long_term'].mean()

    st.metric("전체 평균 6개월 유지율", f"{baseline*100:.1f}%")
    st.metric("주 10시간 이상 유지율", f"{magic_prob*100:.1f}%")
    ''

    # 핵심 행동 변수
    st.header("핵심 행동 변수들")

    features = [
        'ViewingHoursPerWeek',
        'SupportTicketsPerMonth',
        'MonthlyCharges',
        'ContentDownloadsPerMonth',
        'WatchlistSize'
    ]

    corr = df[features + ['Churn']].corr()

    fig3, ax3 = plt.subplots(figsize=(7,5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax3)
    ax3.set_title("Correlation with Churn")
    st.pyplot(fig3)
    # 유저 분화
    st.header("유저 분화")

    palette = {
        0: "#1f77b4",
        1: "#ff7f0e",
        2: "#2ca02c",
        3: "#d62728"
    }

    seg_features = df[
        ['ViewingHoursPerWeek','WatchlistSize',
        'ContentDownloadsPerMonth','SupportTicketsPerMonth']
    ]

    X = StandardScaler().fit_transform(seg_features)
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['segment'] = kmeans.fit_predict(X)

    fig4, ax4 = plt.subplots(figsize=(7,5))
    sns.scatterplot(
        data=df,
        x='ViewingHoursPerWeek',
        y='WatchlistSize',
        hue='segment',
        palette=palette,   
        alpha=0.7,
        ax=ax4
    )
    ax4.set_title("User Segments by Behavior")
    ax4.set_xlabel("Viewing Hours per Week")
    ax4.set_ylabel("Watchlist Size")
    st.pyplot(fig4)
    ''

    # 외부 원인
    st.header(" 외부 원인 ")

    market_df = pd.DataFrame({
        'Reason': ['콘텐츠 부족', '스포츠 부재', '가격 부담'],
        'Percent': [44, 64, 53]
    })

    fig5, ax5 = plt.subplots(figsize=(7,5))
    ax5.plot(market_df['Reason'], market_df['Percent'], marker='o')
    ax5.set_ylim(0,100)
    ax5.set_title("Market Churn Reasons")
    ax5.set_ylabel("%")
    st.pyplot(fig5)
    ''

    # 3개월 무료권 효과
    st.header(" 3개월차 무료권 효과")
    baseline = df['long_term'].mean() 

    risk_group = df[
        (df['tenure'] <= 3) &
        (df['ViewingHoursPerWeek'] < 10)
    ].copy()


    baseline = (risk_group['tenure'] >= 6).mean()

    converted_idx = risk_group.sample(frac=0.4, random_state=42).index

    df_sim = df.copy()
    df_sim.loc[converted_idx, 'tenure'] = 6   

    risk_group_sim = df_sim.loc[risk_group.index]
    improved = (risk_group_sim['tenure'] >= 6).mean()

    x = [0, 1]
    y = [baseline, improved]

    fig6, ax6 = plt.subplots(figsize=(5,4))
    ax6.plot(x, y, marker='o', linewidth=3)
    ax6.set_xticks([0,1])
    ax6.set_xticklabels(['기존','3개월 무료권'])
    ax6.set_xlim(-0.2, 1.2)
    ax6.set_ylim(-0.1, 1)   
    ax6.set_ylabel("6개월 유지 확률")
    ax6.set_title("3개월 무료권 정책 효과 (High-risk Users)")
    ax6.grid(alpha=0.3)
    st.pyplot(fig6)
    ''

    # 스포츠 라이브 도입 효과
    sports_df = pd.DataFrame({
    'Service': ['Netflix','Tving','Coupang'],
    'Live': [0,1,1]
    })

    fig7, ax7 = plt.subplots()
    ax7.plot(sports_df['Service'], sports_df['Live'], marker='o')
    ax7.set_title("Live Sports Availability")
    st.pyplot(fig7)
    ''

    # 결합 상품
    bundle_count_df = pd.DataFrame({
    'Count':['1개','2개','3개','4개','5개'],
    'Ratio':[20,40,23,10,3]
    })

    bundle_brand_df = pd.DataFrame({
        'OTT':['Netflix','Coupang','Tving','Disney+','Wave'],
        'Ratio':[86,52,39,23,16]
    })

    fig8, ax8 = plt.subplots(1,2, figsize=(10,4))
    ax8[0].pie(bundle_count_df['Ratio'], labels=bundle_count_df['Count'], autopct='%1.0f%%')
    ax8[0].set_title("구독 개수 분포")

    ax8[1].pie(bundle_brand_df['Ratio'], labels=bundle_brand_df['OTT'], autopct='%1.0f%%')
    ax8[1].set_title("결합상품 브랜드 구성")
    st.pyplot(fig8)
    ''
    combo_df = pd.DataFrame({
        'Combo': [
            'Netflix + Coupang', 
            'Netflix + Tving', 
            'Netflix + Disney+', 
            'Coupang + Tving', 
            'Netflix + Coupang + Tving'
        ],
        'Ratio': [28, 22, 15, 12, 23]
    })

    fig9, ax9 = plt.subplots()
    ax9.pie(combo_df['Ratio'], 
        labels=combo_df['Combo'], 
        autopct='%1.0f%%')
    ax9.set_title("주요 OTT 결합 조합")
    st.pyplot(fig9)
    ''
    # 최종 요악
    st.header("최종 결론")

    st.markdown(f"""
    ## 최종 결론: OTT Churn은 행동과 구조의 문제다

    ### 시간 구조적 특성
    Kaplan-Meier 생존 분석 결과,  
    OTT 이탈은 무작위적으로 발생하지 않으며  
    **가입 후 약 3개월 시점에서 구조적 이탈 임계구간**이 존재한다.  
    이는 churn이 단순 만족도의 문제가 아니라  
    **시간 의존적 위험 구조(time-dependent risk)**임을 의미한다.

    ### 행동 기반 전환점 (Magic Moment)
    산점도 및 Engagement Score 분석 결과,  
    **주당 약 10시간 이상의 시청량을 넘는 순간  
    사용자는 장기 생존 궤도로 진입**하는 경향을 보였다.  
    이는 특정 행동 임계점(Magic Moment)이  
    생존 확률 구조를 비선형적으로 변화시킴을 시사한다.

    ### 핵심 이탈 요인 (Magic Drivers)
    상관분석 결과,  
    Support Tickets 및 Monthly Charges는 churn과 양의 상관을,  
    Viewing Hours 및 Watchlist Size는 음의 상관을 보였다.  
    즉, 이탈은 감정적 요인이 아니라  
    **콘텐츠 소비 강도와 비용 부담이라는 구조적 변수**에 의해 설명된다.

    ### 사용자 구조 분화
    K-means 군집 분석 결과,  
    사용자는 행동 공간에서 최소 **4개의 구조적 유형으로 분화**되며,  
    churn은 개별 사용자의 성향이 아니라  
    **소속된 행동 군집의 속성에 의해 결정**된다.

    ###  정책 개입 효과
    3개월 무료권 시뮬레이션 결과,  
    고위험군(Low engagement, 초기 사용자)에서  
    장기 유지율이 유의미하게 상승하였다.  
    이는 가격 인센티브가 단순 만족이 아니라  
    **이탈 위험 구조 자체를 이동시키는 정책 수단**임을 의미한다.

    ###  시장 구조적 한계
    시장 데이터 분석 결과,  
    스포츠/라이브 콘텐츠 부재와 가격 부담은  
    OTT churn의 핵심 외부 구조 요인으로 확인되었으며,  
    멀티 OTT 사용은 이미 시장의 기본 상태이다.

    ---

    ## 종합 해석
    OTT churn은 개인 만족도의 문제가 아니라  
    사용자 행동 구조 + 콘텐츠 포트폴리오 + 
    가격 구조가 결합된 시스템적 현상(system-level phenomenon)이다.

    따라서 효과적인 churn 관리 전략은  
    단일 기능 개선이 아니라,  
    **초기 행동 유도 → 콘텐츠 구조 개선 → 가격 개입을 포함한  
    통합적 생존 설계 전략**이어야 한다.
    """)

# Page3: 기존 고객 유지 전략 페이지
elif st.session_state.page =='retention':

    # 뒤로가기 버튼
    if st.button("홈으로 돌아가기"):
        go_to_page('home')

    with header_col1:
        st.image('https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg', width=250)

    with header_col2:
        st.title('기존 고객 유지 전략')
        st.text('기존 고객 유지를 위한 전략 및 이탈 방지 시뮬레이션')

    tab1, tab2, tab3, tab4 = st.tabs(["전략 1: 마케팅 분야", "전략 2: 서비스 모델의 변화", "전략 3: 유통 및 플랫폼 전략","종합 예상 효과"])

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
    with tab4:
        st.markdown('### 종합 예상 효과')
        effect_col1, effect_col2, effect_col3, effect_col4 = st.columns(4)
        with effect_col1:
            st.metric('이탈률 감소', '25-35%', delta='-30%', delta_color='inverse')
        with effect_col2:
            st.metric('평균 구독 기간', '+4개월', delta='+4개월')
        with effect_col3:
            st.metric("고객 LTV", "+40%", delta="+40%")
        with effect_col4:
            st.metric("재가입률", "+50%", delta="+50%")

        st.success("**결론**: 이 전략들을 종합적으로 실행하면 고객 유지율을 크게 향상시키고, 장기적인 수익성을 확보할 수 있습니다.")

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

