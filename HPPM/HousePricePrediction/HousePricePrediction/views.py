from django.shortcuts import render
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline
import matplotlib
matplotlib.rcParams["figure.figsize"]=(20,10)



def home(request):
    return render(request, "home.html")

def predict(request):
    return render(request, "predict.html")

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens)==2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats={}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]={
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')




def result(request):
    df1 = pd.read_csv(r"C:\HPPM\model\Bengaluru_House_Data.csv")
    df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns')
    df3 = df2.dropna()
    df3['size'].unique()
    df3['bhk']=df3['size'].apply(lambda x: int(x.split(' ')[0]))
    #bhk = int(bhk)
    df4 = df3.copy()
    df5 = df4.copy()
    df5['total_sqft'] = pd.to_numeric(df5['total_sqft'],errors='coerce')
    df5['bhk'] = pd.to_numeric(df5['bhk'],errors='coerce')
    df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
    location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
    location_stats_less_than_10 = location_stats[location_stats<=10]
    df5.location=df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
    len(df5.location.unique())
    df6 =df5[~(df5.total_sqft/df5.bhk<300)]

    df7 = remove_pps_outliers(df6)
    df8 = remove_bhk_outliers(df7)
    df9 = df8[df8.bath<df8.bhk+2]
    df10=df9.drop(['size'],axis='columns')
    dummies = pd.get_dummies(df10.location)
    df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
    df12 = df11.drop('location',axis='columns')
    X = df12.drop('price',axis='columns')
    y = df12.price

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
    
    from sklearn.linear_model import LinearRegression
    lr_clf = LinearRegression()
    lr_clf.fit(X_train,y_train)
    lr_clf.score(X_test,y_test)
    

    from sklearn.model_selection import ShuffleSplit
    from sklearn.model_selection import cross_val_score
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    cross_val_score(LinearRegression(),X,y,cv=cv)

    var1 = request.GET['location']
    var2 = float(request.GET['sqft'])
    var3 = float(request.GET['bath'])
    var4 = request.GET['bhk']
    
    def predict_price(location,sqft,bath,bhk):
       loc_index = np.where(X.columns==location)[0][0] 
       x=np.zeros(len(X.columns))
       x[0]= sqft
       x[1]= bath
       x[2]= bhk
       if loc_index>=0:
            x[loc_index]=1
       return lr_clf.predict([x])[0]

    pred = predict_price(var1, var2, var3, var4)
    

    Price = "The predicted price is ₹" + str(abs(round(pred,2))) + " Lacs"
    
    
    return render(request, "predict.html", {"result2":Price})