import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# 한글 폰트 설정 (macos 기준)
plt.rcParams['font.family'] = 'AppleGothic'
st.title('Netflix 구독 이탈 분석')

# csv 파일 읽기
df = pd.read_csv('data/Subscription_Service_Churn_Dataset.csv')

# 마지막 로그인별 이탈률 계산
churn_by_age = df.groupby('AccountAge').agg(
    churned_count=('Churn', 'sum'),
    total_count=('Churn', 'count'),
    churn_rate=('Churn', 'mean')
).reset_index()

churn_by_age['churn_rate'] = churn_by_age['churn_rate']*100
# 차트 1: 이탈률 추이
st.subheader('계정 사용 기간에 따른 이탈률')
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(churn_by_age['AccountAge'], churn_by_age['churn_rate'], 
         marker='o', linewidth=2, markersize=4, color='#FF4B4B')
ax1.set_xlabel('구독 기간 (일)', fontsize=12)
ax1.set_ylabel('이탈률 (%)', fontsize=12)
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

# 차트 2: 이탈 고객 수
st.subheader('계정 사용 기간 별 이탈 고객 수')
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.bar(churn_by_age['AccountAge'], churn_by_age['churned_count'], 
        color='#0068C9')
ax2.set_xlabel('구독 기간 (일)', fontsize=12)
ax2.set_ylabel('이탈 고객 수', fontsize=12)
st.pyplot(fig2)

# 주요 인사이트
st.write('### 주요 인사이트')
max_churn_age = churn_by_age.loc[churn_by_age['churn_rate'].idxmax()]
st.write(f"- 이탈률이 가장 높은 시점: {max_churn_age['AccountAge']:.0f}일 ({max_churn_age['churn_rate']:.1f}%)")

# 전체 평균 이탈률
total_churn_rate = df['Churn'].mean() * 100
st.write(f"- 전체 고객 이탈률: {total_churn_rate:.1f}%")
st.write(f"- 전체 고객 수: {len(df):,}명")
st.write(f"- 이탈 고객 수: {df['Churn'].sum():,}명")

# 데이터 테이블
st.write('### 상세 데이터')
st.dataframe(churn_by_age)
