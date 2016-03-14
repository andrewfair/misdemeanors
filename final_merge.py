

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings, sys

# pd.set_option('display.max_columns', 100)
# warnings.filterwarnings('ignore')
# get_ipython().magic('matplotlib inline')


def main():
  weather = pd.read_csv('../data/vzw.csv')
  env = pd.read_csv('../data/vz311/vz311intersectionsLighter.txt', delimiter='\t', encoding='utf8', engine='python')
  env.columns = [i.replace(' ', '_') for i in env.columns]
  weather['UNIQUE KEY'] = weather['UNIQUE KEY'].fillna(0)
  weather['UNIQUE KEY'] = weather['UNIQUE KEY'].astype(str)
  env['UNIQUE_KEY'] = env.UNIQUE_KEY.fillna(1)
  env['UNIQUE_KEY'] = env.UNIQUE_KEY.astype(str)

  # assign labels
  env['label'] = np.where(env.DATE.isnull(), 0, 1)

  # change the nans to 0s
  env['NUMBER_OF_CYCLIST_INJURED'] = np.where(env.NUMBER_OF_CYCLIST_INJURED.isnull(), 0, env.NUMBER_OF_CYCLIST_INJURED)
  env['NUMBER_OF_CYCLIST_KILLED'] = np.where(env.NUMBER_OF_CYCLIST_KILLED.isnull(), 0, env.NUMBER_OF_CYCLIST_KILLED)
  env['NUMBER_OF_MOTORIST_INJURED'] = np.where(env.NUMBER_OF_MOTORIST_INJURED.isnull(), 0, env.NUMBER_OF_MOTORIST_INJURED)
  env['NUMBER_OF_MOTORIST_KILLED'] = np.where(env.NUMBER_OF_MOTORIST_KILLED.isnull(), 0, env.NUMBER_OF_MOTORIST_KILLED)
  env['NUMBER_OF_PEDESTRIANS_INJURED'] = np.where(env.NUMBER_OF_PEDESTRIANS_INJURED.isnull(), 0, env.NUMBER_OF_PEDESTRIANS_INJURED)
  env['NUMBER_OF_PEDESTRIANS_KILLED'] = np.where(env.NUMBER_OF_PEDESTRIANS_KILLED.isnull(), 0, env.NUMBER_OF_PEDESTRIANS_KILLED)
  env['NUMBER_OF_PERSONS_INJURED'] = np.where(env.NUMBER_OF_PERSONS_INJURED.isnull(), 0, env.NUMBER_OF_PERSONS_INJURED)
  env['NUMBER_OF_PERSONS_KILLED'] = np.where(env.NUMBER_OF_PERSONS_KILLED.isnull(), 0, env.NUMBER_OF_PERSONS_KILLED)

  # assign speical vehicle code type for nans
  codes = np.arange(1,6, dtype=np.int64)
  for i in codes:
      car_type = 'VEHICLE_TYPE_CODE_%d' % i
      env[car_type] = np.where(env[car_type].isnull(), 'CONTROL', env[car_type])
      
  # assign special contributing factor for nans
  for j in codes:
      fac_type = 'CONTRIBUTING_FACTOR_VEHICLE_%d' % j
      env[fac_type] = np.where(env[fac_type].isnull(), 'CONTROL', env[fac_type])

  df = env.merge(weather, how='left', left_on=['UNIQUE_KEY'], right_on=['UNIQUE KEY'])

  fw = pd.read_csv('../data/weather_raw_NYC/full_weather.csv') 

  np.random.seed(83)
  scores = np.random.random_integers(0, fw.shape[0]-1, size=df.DATE_x.isnull().sum())

  eidx = df[df.DATE_x.isnull()].index.values

  gg = fw.iloc[scores]

  gg['new_index'] = eidx
  gg.columns = ['new_%s' % j for j in gg.columns]

  df2 = df.merge(gg, how='left', left_index=True, right_on='new_new_index')

  # replace the nan values for null dates in full file
  df2['Visibility'] = np.where(df2.DATE_x.isnull(), df2.new_Visibility, df2.Visibility)
  df2['WetBulbFarenheit'] = np.where(df2.DATE_x.isnull(), df2.new_WetBulbFarenheit, df2.WetBulbFarenheit)
  df2['WindSpeed'] = np.where(df2.DATE_x.isnull(), df2.new_WindSpeed, df2.WindSpeed)
  df2['Precip'] = np.where(df2.DATE_x.isnull(), df2.new_Precip, df2.Precip)
  df2['PrecipSum'] = np.where(df2.DATE_x.isnull(), df2.new_PrecipSum, df2.PrecipSum)

  # df = pd.read_csv('../data/vz_full.csv')
  df2['identity'] = ['id-%d' % d for d in np.arange(0, df2.shape[0], dtype=np.int64)]

  # subset the columns
  cc = ['identity', 'label', 'NUMBER_OF_PERSONS_INJURED', 'NUMBER_OF_PERSONS_KILLED',
        'NUMBER_OF_PEDESTRIANS_INJURED', 'NUMBER_OF_PEDESTRIANS_KILLED','NUMBER_OF_CYCLIST_INJURED', 
        'NUMBER_OF_CYCLIST_KILLED','NUMBER_OF_MOTORIST_INJURED', 'NUMBER_OF_MOTORIST_KILLED',
        'CONTRIBUTING_FACTOR_VEHICLE_1', 'CONTRIBUTING_FACTOR_VEHICLE_2', 'CONTRIBUTING_FACTOR_VEHICLE_3',
        'CONTRIBUTING_FACTOR_VEHICLE_4', 'CONTRIBUTING_FACTOR_VEHICLE_5', 'VEHICLE_TYPE_CODE_1',
        'VEHICLE_TYPE_CODE_2', 'VEHICLE_TYPE_CODE_3', 'VEHICLE_TYPE_CODE_4', 'VEHICLE_TYPE_CODE_5',
        'Street_Condition', 'Traffic_Signal_Condition', 'Visibility', 'WetBulbFarenheit',
        'WindSpeed', 'Precip', 'PrecipSum','intersectionID']

  # subset
  d1 = df2[cc]

  # remove the column spaces with _
  d1.columns = [x.replace(' ', '_') for x in d1.columns]

  # total persons involved in accident [injured + killed]
  d1['total_involved'] = np.sum(d1.values[:,1:9].astype(np.float64), axis=1).ravel()

  # d1.to_csv('temp_data.csv')
  d1 = pd.read_csv('temp_data.csv').drop(['Unnamed: 0'],1)

  # dummy out the strings
  d2 = pd.DataFrame(d1.identity)
  for head in d1.columns:
      if head != 'identity':
          if d1[head].dtype == np.float64:
              d2 = d2.merge(pd.DataFrame(d1[head]), left_index=True, right_index=True)
          elif d1[head].dtype == object:
              temp = pd.get_dummies(d1[head], prefix=head, prefix_sep='-', dummy_na=True)
              d2 = d2.merge(temp, right_index=True, left_index=True)
          elif d1[head].dtype == np.int64:
              d2[head] =d1[head]
          else:
              print('Skipped a Column', head)

if __name__ == '__main__':
  main()
