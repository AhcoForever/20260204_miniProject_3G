import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 한글 폰트 설정 (Windows)
font_path = "/Users/kmy/Library/Fonts/Pretendard-Light.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)

# 마이너스 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter

st.set_page_config(layout="wide")
st.title("OTT Churn Analysis: 현황 → 원인 → 전략")

# =========================
# 1. 데이터 로드
# =========================
df = pd.read_csv("data/Subscription_Service_Churn_Dataset.csv")

# 핵심 변수
df['tenure'] = df['AccountAge']
df['churn'] = df['Churn']
df['long_term'] = df['tenure'] >= 6

# =========================
# A. 현황: 3개월 이탈 패턴
# =========================
st.header("1. 현황: 가입 후 3개월 이탈 급증")

kmf = KaplanMeierFitter()
kmf.fit(df['tenure'], event_observed=df['churn'])

fig1, ax1 = plt.subplots(figsize=(7,5))
kmf.plot_survival_function(ax=ax1)
ax1.axvline(3, color='red', linestyle='--')
ax1.set_title("Survival Curve (Churn over Time)")
ax1.set_xlabel("Months")
ax1.set_ylabel("Survival Probability")
st.pyplot(fig1)

st.markdown("""
**해석**  
가입 후 약 **3개월 시점에서 생존 확률이 급격히 감소**.  
→ 대부분의 이탈은 *초기 3개월 내* 발생.
""")

# =========================
# B. 원인 1: 내부 행동 요인
# =========================
st.header("2-1. 내부 원인: 행동 패턴")

df['user_type'] = np.where(
    df['ViewingHoursPerWeek'] >= df['ViewingHoursPerWeek'].median(),
    'Heavy User', 'Light User'
)

fig2, ax2 = plt.subplots(figsize=(7,5))
sns.lineplot(
    data=df,
    x='tenure',
    y='ViewingHoursPerWeek',
    hue='user_type',
    estimator='mean',
    ax=ax2
)
ax2.set_title("Customer Journey: Heavy vs Light")
ax2.set_xlabel("Account Age (Months)")
ax2.set_ylabel("Viewing Hours / Week")
st.pyplot(fig2)

st.markdown("""
**해석**  
Light User는 **2~3개월부터 시청량 급락 → churn**.  
Heavy User는 시청 유지 → 생존.
""")

# =========================
# B. 원인 2: 외부 구조 요인 (PDF 기반)
# =========================
st.header("2-2. 외부 원인: 시장 구조 (PDF)")

market_df = pd.DataFrame({
    'Reason': ['볼 콘텐츠 부족', '스포츠/라이브 부재', '가격 부담'],
    'Percent': [44, 64, 53]
})

fig3, ax3 = plt.subplots(figsize=(7,5))
sns.barplot(data=market_df, x='Reason', y='Percent', ax=ax3)
ax3.set_title("Why Users Churn (Market Survey)")
ax3.set_ylabel("응답 비율 (%)")
st.pyplot(fig3)

st.markdown("""
**해석**  
넷플릭스는 **스포츠/라이브 콘텐츠 부재** → 구조적 약점.  
가격 부담도 주요 이탈 요인.
""")

# =========================
# C. 전략 1: 3개월 무료권 효과
# =========================
st.header("3-1. 전략: 3개월차 무료권 개입")

baseline = df['long_term'].mean()

# 가정: 3개월에 무료권 제공 시 churn 20% 감소
improved_prob = baseline + 0.20

fig4, ax4 = plt.subplots(figsize=(5,4))
ax4.bar(['기존', '무료권 개입'], [baseline, improved_prob])
ax4.set_ylim(0,1)
ax4.set_title("Retention Improvement Simulation")
ax4.set_ylabel("6개월 생존 확률")
st.pyplot(fig4)

# =========================
# C. 전략 2: 스포츠 도입 효과
# =========================
st.header("3-2. 전략: 스포츠/라이브 도입")

sports_df = pd.DataFrame({
    'Service': ['Netflix', 'Tving', 'Coupang Play'],
    'LiveContent': [0, 1, 1]
})

fig5, ax5 = plt.subplots(figsize=(5,4))
sns.barplot(data=sports_df, x='Service', y='LiveContent', ax=ax5)
ax5.set_title("Live Content Availability")
ax5.set_ylabel("Live 제공 여부")
st.pyplot(fig5)

st.markdown("""
**해석**  
경쟁사 대비 Netflix는 **라이브 콘텐츠 구조적으로 불리**.  
→ 스포츠 도입은 churn 감소의 구조적 전략.
""")

# =========================
# C. 전략 3: 결합상품 효과
# =========================
st.header("3-3. 전략: 결합상품")

bundle_df = pd.DataFrame({
    '구독 개수': ['1개', '2개', '3개 이상'],
    '비율': [28, 45, 27]
})

fig6, ax6 = plt.subplots(figsize=(5,4))
ax6.pie(bundle_df['비율'], labels=bundle_df['구독 개수'], autopct='%1.0f%%')
ax6.set_title("Multi-OTT Subscription")
st.pyplot(fig6)

st.markdown("""
**해석**  
대다수 사용자는 이미 **2개 이상 OTT 사용**.  
→ 단독 상품보다 **결합상품이 구조적으로 유리**.
""")

# =========================
# 최종 전략 요약
# =========================
st.header("최종 결론: 데이터 기반 전략")

st.markdown("""
### 현황
- 가입 후 **3개월 시점 이탈 집중**
- Light User는 시청량 급락 후 churn

### 원인
- 내부: 콘텐츠 소비 감소
- 외부: 스포츠/라이브 부재, 가격 부담

### 전략
1. **3개월차 무료권 자동 제공**
2. **스포츠/라이브 콘텐츠 도입**
3. **통신사/플랫폼 결합상품**

> OTT 이탈은 개인 만족 문제가 아니라  
> **콘텐츠 포트폴리오 + 가격 구조 문제**다.
""")
