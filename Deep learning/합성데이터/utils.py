
import numpy as np
import pandas as pd

# for continuous variables
def calculate_ks_complement(df1, df2, column):
    ks_statistic, _ = ks_2samp(df1[column], df2[column])
    ks_complement = 1 - ks_statistic
    return ks_complement

# for categorical variables
def calculate_tv_complement(df1, df2, column):
    df1_counts = df1[column].value_counts(normalize=True)
    df2_counts = df2[column].value_counts(normalize=True)
    tv_distance = sum(abs(df1_counts.get(key, 0) - df2_counts.get(key, 0)) for key in set(df1_counts.keys()).union(df2_counts.keys()))
    tv_complement = 1 - tv_distance / 2
    return tv_complement

# to compare the shapes of columns in two dataframes
def compare_column_shapes(df1, df2):
    common_columns = set(df1.columns).intersection(df2.columns)
    total_columns = len(common_columns)
    if total_columns == 0:
        return 0.0  # No common columns

    shape_similarity_score = 0
    for col in common_columns:
        # Check if column is numeric for KS Test, otherwise use TV for categorical data
        if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
            shape_similarity_score += calculate_ks_complement(df1, df2, col)
        else:
            shape_similarity_score += calculate_tv_complement(df1, df2, col)

    return shape_similarity_score / total_columns

# Function to compare the trends of column pairs in two dataframes
def compare_column_pair_trends(df1, df2):
    common_columns = list(set(df1.columns).intersection(df2.columns))
    total_pairs = len(common_columns) * (len(common_columns) - 1) // 2
    if total_pairs == 0:
        return 0.0  # No pairs to compare

    trend_similarity_score = 0
    for i in range(len(common_columns)):
        for j in range(i + 1, len(common_columns)):
            col1, col2 = common_columns[i], common_columns[j]

            # Compute correlations only for numeric columns
            if pd.api.types.is_numeric_dtype(df1[col1]) and pd.api.types.is_numeric_dtype(df2[col1]) \
               and pd.api.types.is_numeric_dtype(df1[col2]) and pd.api.types.is_numeric_dtype(df2[col2]):
                corr1 = df1[[col1, col2]].corr().iloc[0, 1]
                corr2 = df2[[col1, col2]].corr().iloc[0, 1]
                # Using 1 - absolute difference to get a similarity measure
                trend_similarity_score += 1 - abs(corr1 - corr2)

    return trend_similarity_score / total_pairs if total_pairs > 0 else 0.0

# GU
def check_uniqueness(origin_data, synthetic_data, lst_quasi_identifier):
  unique_origin_data = origin_data.drop_duplicates(subset=lst_quasi_identifier)
  unique_origin_data_qi = unique_origin_data[ lst_quasi_identifier ]

  unique_synthetic_data = synthetic_data.drop_duplicates(subset=lst_quasi_identifier)
  unique_synthetic_data_qi = unique_synthetic_data[ lst_quasi_identifier ]

  denominator = len(unique_origin_data_qi)
  numerator = 0
  val_uniqueness = 0

  if denominator == 0:
    print('uniqueness(0.0~1.0) : ', val_uniqueness)
    return val_uniqueness

  for i in unique_origin_data_qi.values:
     for j in unique_synthetic_data_qi.values:
       if np.array_equal(i,j):
         numerator += 1
         break

  print(denominator)
  print(numerator)

  if denominator == 0:
    return val_uniqueness

  val_uniqueness = numerator / denominator

  print('uniqueness(0.0~1.0) : ', val_uniqueness)
  return val_uniqueness



# TCAP (itertuples 방식)

def WEAP_data1( synthetic_data, lst_key_variables, lst_target_variables, adjust_alpha_val_WEAP_j):
  synthetic_data_WEAP_1 = synthetic_data.copy()
  #key_var_synthetic_data = synthetic_data[ lst_key_variables ]
  #target_var_synthetic_data = synthetic_data[ lst_target_variables ]

  denominator = 0
  numerator = 0

  for series_j in synthetic_data.itertuples():
    denominator = 0
    numerator = 0

    for series_i in synthetic_data.itertuples():
      #print(series_i.Index)
      if series_i.Index == series_j.Index:
        continue
      elif series_j.Education == series_j.Education and series_j.Marital_Status == series_j.Marital_Status:
      #elif series_j.education == series_i.education and series_j.workclass == series_i.workclass and series_j.occupation == series_i.occupation and series_j.race == series_i.race :
        denominator +=1
        if series_j.Year_Birth == series_j.Year_Birth :
        #if series_j.age == series_i.age and series_j.sex == series_i.sex :
          numerator +=1

    if denominator == 0:
      #print('분모 0 인 idx_j : ', idx_j)
      synthetic_data_WEAP_1.drop( [series_j.Index], axis=0, inplace=True)
      continue


    WEAP_j = numerator / denominator
    #print('idx_j : ', idx_j)
    #print('WEAP_j : ' , WEAP_j)
    if WEAP_j < adjust_alpha_val_WEAP_j:
      synthetic_data_WEAP_1.drop( [series_j.Index], inplace=True)
    #else:
    #  print('리스크한 레코드로 선정( WEAP_j >= adjust_alpha_val_WEAP_j )')
    #  print('idx_j: ', idx_j)

  return synthetic_data_WEAP_1

  # return : WEAP 가 1인 합성데이터 데이터프레임 전체

def TCAP_data1(origin_data, synthetic_data_WEAP_alpha, key_lst_syn, target_lst_syn):
  lst_TCAP_j = [] # return은 평균이어야 함

  denominator = 0
  numerator = 0

  for origin_series_j in origin_data.itertuples():
    denominator = 0
    numerator = 0
    #print(origin_series_j)
    for syn_series_j in synthetic_data_WEAP_alpha.itertuples():
      if origin_series_j.Education == syn_series_j.Education and origin_series_j.Marital_Status == syn_series_j.Marital_Status:
      #if origin_series_j.education == syn_series_j.education and origin_series_j.workclass == syn_series_j.workclass and origin_series_j.occupation == syn_series_j.occupation and origin_series_j.race == syn_series_j.race :
        denominator +=1
        if origin_series_j.Year_Birth == syn_series_j.Year_Birth:
        #if origin_series_j.age == syn_series_j.age and origin_series_j.sex == syn_series_j.sex:
          numerator +=1

    if denominator == 0:
      #print('분모 0 인 origin data의 Index : ', origin_series_j.Index)
      continue


    TCAP_j = numerator / denominator
    #print('TCAP_j : ' , TCAP_j)
    lst_TCAP_j.append(TCAP_j)
    #else:
    #  print('리스크한 레코드로 선정( WEAP_j >= adjust_alpha_val_WEAP_j )')
    #  print('idx_j: ', idx_j)

  return sum(lst_TCAP_j) / len(lst_TCAP_j)
  # return : score 점수 (TCAP_j 의 평균)


def WEAP_data2( synthetic_data, lst_key_variables, lst_target_variables, adjust_alpha_val_WEAP_j):
  synthetic_data_WEAP_1 = synthetic_data.copy()
  #key_var_synthetic_data = synthetic_data[ lst_key_variables ]
  #target_var_synthetic_data = synthetic_data[ lst_target_variables ]

  denominator = 0
  numerator = 0

  for series_j in synthetic_data.itertuples():
    denominator = 0
    numerator = 0

    for series_i in synthetic_data.itertuples():
      #print(series_i.Index)
      if series_i.Index == series_j.Index:
        continue
      #elif series_j.Education == series_j.Education and series_j.Marital_Status == series_j.Marital_Status:
      elif series_j.education == series_i.education and series_j.workclass == series_i.workclass and series_j.occupation == series_i.occupation and series_j.race == series_i.race :
        denominator +=1
        #if series_j.Income == series_j.Income and series_j.Year_Birth == series_j.Year_Birth :
        if series_j.age == series_i.age :
          numerator +=1

    if denominator == 0:
      #print('분모 0 인 idx_j : ', idx_j)
      synthetic_data_WEAP_1.drop( [series_j.Index], axis=0, inplace=True)
      continue


    WEAP_j = numerator / denominator
    #print('idx_j : ', idx_j)
    #print('WEAP_j : ' , WEAP_j)
    if WEAP_j < adjust_alpha_val_WEAP_j:
      synthetic_data_WEAP_1.drop( [series_j.Index], inplace=True)
    #else:
    #  print('리스크한 레코드로 선정( WEAP_j >= adjust_alpha_val_WEAP_j )')
    #  print('idx_j: ', idx_j)

  return synthetic_data_WEAP_1

  # return : WEAP 가 1인 합성데이터 데이터프레임 전체

def TCAP_data2(origin_data, synthetic_data_WEAP_alpha, key_lst_syn, target_lst_syn):
  lst_TCAP_j = [] # return은 평균이어야 함

  denominator = 0
  numerator = 0

  for origin_series_j in origin_data.itertuples():
    denominator = 0
    numerator = 0
    #print(origin_series_j)
    for syn_series_j in synthetic_data_WEAP_alpha.itertuples():
      #if origin_series_j.Education == syn_series_j.Education and origin_series_j.Marital_Status == syn_series_j.Marital_Status:
      if origin_series_j.education == syn_series_j.education and origin_series_j.workclass == syn_series_j.workclass and origin_series_j.occupation == syn_series_j.occupation and origin_series_j.race == syn_series_j.race :
        denominator +=1
        #if origin_series_j.Year_Birth == syn_series_j.Year_Birth and origin_series_j.Income == syn_series_j.Income:
        if origin_series_j.age == syn_series_j.age:
          numerator +=1

    if denominator == 0:
      #print('분모 0 인 origin data의 Index : ', origin_series_j.Index)
      continue


    TCAP_j = numerator / denominator
    #print('TCAP_j : ' , TCAP_j)
    lst_TCAP_j.append(TCAP_j)
    #else:
    #  print('리스크한 레코드로 선정( WEAP_j >= adjust_alpha_val_WEAP_j )')
    #  print('idx_j: ', idx_j)

  return sum(lst_TCAP_j) / len(lst_TCAP_j)
  # return : score 점수 (TCAP_j 의 평균)




def generate_synthetic_data_based_Statistical_Methodology(csv_file, num_samples):
    df = csv_file

    # 빈 DataFrame 생성
    synthetic_df = pd.DataFrame()

    # 각 컬럼에 대해 합성 데이터 생성
    for column in df.columns:
        if df[column].dtype == 'object':  # 범주형 변수 처리
            # 각 범주의 확률 분포 계산
            probs = df[column].value_counts(normalize=True)
            categories = probs.index
            probabilities = probs.values

            # 확률 분포에 따라 랜덤 샘플 생성
            synthetic_df[column] = np.random.choice(categories, size=num_samples, p=probabilities)
        else:  # 연속형 변수 처리
            # 평균과 표준편차 추출
            mean = df[column].mean()
            std = df[column].std()

            # 정규 분포에서 랜덤 샘플 생성
            synthetic_df[column] = np.random.normal(mean, std, num_samples)

    return synthetic_df