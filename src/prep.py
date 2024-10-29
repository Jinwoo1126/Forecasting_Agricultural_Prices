import pandas as pd


class Fluctuation_Probability:
    def __init__(self, df, config):
        self.df = df
        self.items = df['품목명'].unique()
        self.k = config['fluctuation_probability']['k']

    def get_fluctuation_probability(self):
        prob_dict = {}
        for item in self.items:
            prob_dict[item] = {}

            target = self.df[self.df['품목명'] == item]

            
            target['년도'] = target['시점'].str[:4]
            target['월'] = target['시점'].str[4:6]
            target['분기'] = target['월'].astype(int).apply(lambda x: (x - 1) // 3 + 1)

            target['이전_월_가격'] = target['평균가격(원)'].shift(3)  
            target['20%_이상_상승'] = ((target['평균가격(원)'] - target['이전_월_가격']) / target['이전_월_가격'] >= 0.2).astype(int)  

            monthly_counts = target.groupby('월')['20%_이상_상승'].sum()  
            monthly_counts_total = target.groupby('월').size()  
            monthly_lambda_risep = monthly_counts / monthly_counts_total  

            prob_dict[item]['risep'] = monthly_lambda_risep

            target['이전_월_가격'] = target['평균가격(원)'].shift(3)  
            target['20%_이상_하락'] = ((target['평균가격(원)'] - target['이전_월_가격']) / target['이전_월_가격'] <= -0.2).astype(int)  

            monthly_counts = target.groupby('월')['20%_이상_하락'].sum()  
            monthly_counts_total = target.groupby('월').size()  
            monthly_lambda_dropp = monthly_counts / monthly_counts_total  

            prob_dict[item]['dropp'] = monthly_lambda_dropp

            monthly_counts = target.groupby('월').apply(lambda x: (x['평균가격(원)'] > x['평년 평균가격(원)']).sum())
            monthly_counts_total = target.groupby('월').size()

            monthly_lambda = monthly_counts / monthly_counts_total

            quarterly_counts = target.groupby('분기').apply(lambda x: (x['평균가격(원)'] > x['평년 평균가격(원)']).sum())
            quarterly_counts_total = target.groupby('분기').size()

            quarterly_lambda = quarterly_counts / quarterly_counts_total

            prob_dict[item]['high'] = monthly_lambda
            prob_dict[item]['high_quarter'] = quarterly_lambda
        
        return prob_dict