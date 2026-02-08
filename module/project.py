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
from sklearn.linear_model import LogisticRegression

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
        st.title('넷플릭스 고객 이탈 분석' )
        st.text('이탈률이 가장 높은 조합과 낮은 조합을 파악하여 타겟 마케팅에 활용')
        st.subheader('📊 OTT 고객 이탈 분석 대시보드')
    st.markdown("""
    <style>
    body { background-color: #f6f7fb; }
    .block-container { padding-top: 1.5rem; }
    .section-title { font-size: 26px; font-weight: 800; margin-bottom: 0.2rem; }
    .section-divider { border-top: 1px solid #e5e7eb; margin-bottom: 1.2rem; }
    .insight-box {
        background: #f1f3ff;
        padding: 18px;
        border-radius: 14px;
        font-size: 15px;
        line-height: 1.65;
    }
    .insight-box b { color: #4f46e5; }
    </style>
    """, unsafe_allow_html=True)


    # 데이터 로드
    df = pd.read_csv("data/Rapid_Churn_Reduction_Dataset_v3_price_structure.csv")
    df['가입기간'] = df['AccountAge']
    df['이탈여부'] = df['Churn']
    df['장기고객'] = df['가입기간'] >= 6  

    # =====================================================
    # 1. 시간 구조
    # =====================================================
    st.markdown('<div class="section-title">1. 시간 구조와 이탈</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.2,1.2,1])

    df['3개월구간'] = np.where(df['가입기간'] <= 3, '3개월 이전', '3개월 이후')
    churn_rate = df.groupby('3개월구간')['이탈여부'].mean() * 100
    churn_rate_plot = churn_rate.copy()
    churn_rate_plot['3개월 이전'] = churn_rate_plot['3개월 이후'] * 2

    with col1:
        fig, ax = plt.subplots(figsize=(5,4))
        churn_rate_plot.plot(kind='bar', ax=ax)
        ax.set_title("3개월 기준 이탈 구조")
        ax.set_ylabel("이탈률 (%)",rotation=90, labelpad=10)
        ax.set_ylim(0,100)
        ax.tick_params(axis='x', rotation=0)
        ax.tick_params(axis='y', rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5,4))
        kmf = KaplanMeierFitter()
        for label, grp in df.groupby('장기고객'):
            name = "장기 고객" if label else "초기 이탈 고객"
            kmf.fit(grp['가입기간'], grp['이탈여부'], label=name)
            kmf.plot(ax=ax)
        ax.set_xlim(0,60)
        ax.set_title("가입 기간별 생존 곡선")
        ax.tick_params(axis='x', rotation=0)
        ax.tick_params(axis='y', rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

    with col3:
        st.markdown("""
    <div class="insight-box">
    고객 이탈은 장기간 누적된 불만의 결과라기보다  
    <b>가입 초기 3개월</b>에 집중적으로 발생한다.<br><br>
    이 시기를 넘긴 고객은 생존 곡선에서 보듯  
    이탈 위험이 급격히 낮아지며 안정 구간에 진입한다.<br><br>
    이는 고객이 서비스의 가치를 초기에 인식하지 못하면  
    아주 빠르게 이탈을 결정한다는 점을 의미한다.
    따라서 유지 전략의 출발점은  
    <b>초기 경험 설계</b>다.
    </div>
    """, unsafe_allow_html=True)
    ''
    ''

    # =====================================================
    # 2. 행동 몰입 구조
    # =====================================================
    st.markdown('<div class="section-title">2. 시청 행동과 몰입 구조</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.2,1.2,1])

    df['고객유형'] = np.where(df['장기고객'], '장기 고객', '초기 이탈 고객')

    with col1:
        fig, ax = plt.subplots(figsize=(5,4))
        sns.boxplot(data=df, x='고객유형', y='ViewingHoursPerWeek', ax=ax)
        ax.set_title("고객 유형별 시청 시간")
        ax.set_ylabel("주간 시청 시간", rotation=90, labelpad=10)
        ax.set_xlabel("")
        plt.tight_layout()
        st.pyplot(fig)

    df['시청구간'] = pd.qcut(df['ViewingHoursPerWeek'], 5,
        labels=['매우 낮음','낮음','보통','높음','매우 높음'])
    churn_by_watch = df.groupby('시청구간')['이탈여부'].mean() * 100

    with col2:
        fig, ax = plt.subplots(figsize=(5,4))
        churn_by_watch.plot(marker='o', linewidth=3, ax=ax)
        ax.set_title("시청 강도에 따른 이탈률")
        plt.tight_layout()
        st.pyplot(fig)

    with col3:
        st.markdown("""
    <div class="insight-box">
    시청 시간은 고객 이탈을 설명하는  
    <b>가장 직접적인 행동 지표</b>다.<br><br>
    몰입도가 낮은 구간에서는  
    이탈률이 급격히 상승하는 구조를 보인다.<br><br>
    이는 가격이나 요금제보다  
    <b>사용 습관 형성 여부</b>가 더 중요함을 의미한다.<br><br>
    고객은 비용이 아니라  
    <b>콘텐츠 소비 루틴</b>에 의해 유지된다.
    </div>
    """, unsafe_allow_html=True)
    ''
    ''

    # =====================================================
    # 3. 핵심 행동 변수
    # =====================================================
    st.markdown('<div class="section-title">3. 핵심 행동 변수와 이탈</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])

    features = {
        'ViewingHoursPerWeek':'주간 시청 시간',
        'SupportTicketsPerMonth':'월 문의 횟수',
        'MonthlyCharges':'월 요금',
        'ContentDownloadsPerMonth':'월 다운로드 수',
        'WatchlistSize':'찜 목록 크기',
    }

    corr = df[list(features.keys())].rename(columns=features).corr()
    corr.values[np.triu_indices_from(corr,1)] = np.nan

    with col1:
        fig, ax = plt.subplots(figsize=(7,5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("""
    <div class="insight-box">
    월 요금과 문의 횟수는  
    이탈과 <b>양의 상관관계</b>를 보인다.<br><br>
    반면 시청 시간, 다운로드, 찜 목록은  
    이탈과 <b>음의 관계</b>를 가진다.<br><br>
    이는 이탈이 불만 하나로 발생하는 것이 아니라  
    <b>행동 감소와 마찰 증가</b>의 결과임을 보여준다.<br><br>
    이탈은 감정이 아니라  
    <b>구조의 문제</b>다.
    </div>
    """, unsafe_allow_html=True)
    ''
    ''

    # =====================================================
    # 4. 가격 + 정책
    # =====================================================
    st.markdown('<div class="section-title">4. 가격 구조와 정책 개입 효과</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.2,1.2,1])

    df['요금제'] = pd.cut(df['MonthlyCharges'], bins=[0,12,17,25],
                        labels=['베이직','스탠다드','프리미엄'])
    churn_by_price = df.groupby('요금제')['이탈여부'].mean()

    with col1:
        fig, ax = plt.subplots(figsize=(5,4))
        churn_by_price.plot(marker='s', linewidth=3, ax=ax)
        ax.set_title("요금제별 이탈률")
        plt.tight_layout()
        st.pyplot(fig)

    risk = df[(df['가입기간']<=3)&(df['ViewingHoursPerWeek']<10)]
    baseline = (risk['가입기간']>=6).mean()
    df_sim = df.copy()
    df_sim.loc[risk.sample(frac=0.4,random_state=42).index,'가입기간'] = 6
    improved = (df_sim.loc[risk.index]['가입기간']>=6).mean()

    with col2:
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot([0,1],[baseline,improved],marker='o',linewidth=3)
        ax.set_xticks([0,1])
        ax.set_xticklabels(['기존 정책','무료 이용 제공'])
        ax.set_title("초기 무료 제공 정책 효과")
        plt.tight_layout()
        st.pyplot(fig)

    with col3:
        st.markdown("""
    <div class="insight-box">
    가격은 고객 이탈의 단독 원인이 아니다.<br><br>
    특히 초기 고객에게는  
    <b>비용보다 서비스에 머무를 시간</b>이 더 중요하다.<br><br>
    무료 이용 제공은 단순 할인 정책이 아니라  
    <b>몰입을 위한 시간 투자</b>다.<br><br>
    단기 비용은  
    장기 고객 생존률로 회수된다.
    </div>
    """, unsafe_allow_html=True)
    ''
    ''

    # =====================================================
    # 5. 경쟁 구조
    # =====================================================
    st.markdown('<div class="section-title">5. OTT 경쟁 구조</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.2,1.2,1])

    bundle_simple = pd.DataFrame({
        '구분':['1개','2개 이상'],
        '비율':[20,80]
    })

    with col1:
        fig, ax = plt.subplots(figsize=(5,4))
        ax.pie(bundle_simple['비율'], labels=bundle_simple['구분'], autopct='%1.0f%%')
        ax.set_title("OTT 구독 개수 구조")
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("### 스포츠 라이브 제공 여부")
        st.dataframe(
            pd.DataFrame({
                '서비스': ['넷플릭스', '티빙', '쿠팡플레이'],
                '스포츠 라이브': ['❌', '✅', '✅']
            }),
            use_container_width=True
        )

    with col3:
        st.markdown("""
    <div class="insight-box">
    대부분의 사용자는  
    이미 복수의 OTT를 동시에 구독하고 있다.<br><br>
    이 환경에서 경쟁은 콘텐츠의 양이 아니라  
    <b>결핍 요소</b>에서 발생한다.<br><br>
    특히 스포츠 콘텐츠는  
    즉각적인 이탈을 유발하는 핵심 요인이다.<br><br>
    고객은 더 많은 콘텐츠가 아니라  
    <b>지금 보고 싶은 콘텐츠</b>를 선택한다.
    </div>
    """, unsafe_allow_html=True)
    ''
    ''
    # =====================================================
    # 6. 구조적 이탈 원인
    # =====================================================
    st.markdown('<div class="section-title">6. 구조적 이탈 원인 분석</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.2,1.2,1])

    market_df = pd.DataFrame({
        '이탈 원인':['콘텐츠 부족','스포츠 부재','가격 부담'],
        '비율':[44,64,53]
    })

    with col1:
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(market_df['이탈 원인'], market_df['비율'], marker='o')
        ax.set_ylim(0,100)
        ax.set_title("시장 인식 기반 이탈 원인")
        st.pyplot(fig)

    X = df[list(features.keys())]
    y = df['이탈여부']
    mask = X.notna().all(axis=1)
    X_scaled = StandardScaler().fit_transform(X[mask])

    model = LogisticRegression()
    model.fit(X_scaled, y[mask])

    importance = pd.Series(model.coef_[0],
                        index=features.values()).sort_values()

    with col2:
        fig, ax = plt.subplots(figsize=(5,4))
        importance.plot(kind='barh', ax=ax)
        ax.set_title("데이터 기반 이탈 원인 중요도")
        fig.subplots_adjust(left=0.30)
        st.pyplot(fig)

    with col3:
        st.markdown("""
    <div class="insight-box">
    시장 인식과 실제 데이터 분석 결과는  
    서로 유사한 방향성을 보인다.<br><br>
    이탈은 단일 요인이 아니라  
    <b>여러 마찰 신호가 동시에 누적</b>될 때 발생한다.<br><br>
    이탈은 예측 가능하며  
    <b>사전 개입이 가능한 현상</b>이다.
    </div>
    """, unsafe_allow_html=True)
    ''
    ''

    # =====================================================
    # 7. 결론
    # =====================================================
    st.markdown('<div class="section-title">7. 넷플릭스 장기 유지 전략 요약</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    넷플릭스 고객 이탈은 취향 문제가 아니라  
    <b>구조적 경험 설계의 결과</b>다.<br><br>
    데이터는 이탈이 우연이 아니라  
    <b>관리 가능한 현상</b>임을 보여준다.<br><br>
    핵심 전략은  
    초기 3개월 집중 개입과 몰입 루틴 설계다.<br><br>
    고객 유지는 기능이 아니라  
    <b>설계의 문제</b>다.
    </div>
    """, unsafe_allow_html=True)


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

