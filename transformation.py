from __future__ import print_function, division

from pandas.api.types import is_numeric_dtype, is_bool_dtype
import pandas as pd
from sklearn.preprocessing import *
from utilities import _randint, _randchoice, _randuniform

pd.options.mode.chained_assignment = None


def new_preprocessor(preprocessor):

    # [standard_scaler, minmax_scaler, maxabs_scaler, [robust_scaler] * 2, kernel_centerer,
    #  [quantile_transform] * 2, normalizer, [binarize] * 2]

    if isinstance(preprocessor, StandardScaler):
        return StandardScaler()
    elif isinstance(preprocessor, MinMaxScaler):
        return MinMaxScaler()
    elif isinstance(preprocessor, MaxAbsScaler):
        return MaxAbsScaler()
    elif isinstance(preprocessor, RobustScaler):
        return RobustScaler()
    elif isinstance(preprocessor, KernelCenterer):
        return KernelCenterer()
    elif isinstance(preprocessor, QuantileTransformer):
        return QuantileTransformer()
    elif isinstance(preprocessor, Normalizer):
        return Normalizer()
    elif isinstance(preprocessor, Binarizer):
        return Binarizer()
    else:
        float("unknown preprocessor")


def nomalization(df, methods='min-max'):
    # nomalization the preprocessed columns from dataset
    # methods contain mean, and min-max.
    for c in df.columns:
        if is_numeric_dtype(df[c]) and is_bool_dtype(df[c])!=True:
            pd.to_numeric(df[c], downcast='float')
            if methods == 'mean':
                df[c] = (df[c] - df[c].mean()) / df[c].std()
            if methods == 'min-max':
                df[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())
    return df

def standard_scaler():
    scaler = StandardScaler()
    return scaler, StandardScaler.__name__

def minmax_scaler():
    scaler = MinMaxScaler()
    return scaler, MinMaxScaler.__name__

def maxabs_scaler():
    scaler = MaxAbsScaler()
    return scaler, MaxAbsScaler.__name__


## IQR parameter
def robust_scaler():
    a,b=_randint(0,50),_randint(51,100)
    scaler = RobustScaler(quantile_range=(a, b))
    tmp=str(a)+"_"+str(b)+"_"+RobustScaler.__name__
    return scaler, tmp

def kernel_centerer():
    scaler = KernelCenterer()
    return scaler, KernelCenterer.__name__

## Tunable parameters
def quantile_transform():
    a, b = _randint(100, 1000), _randint(1000, 1e5)
    c=_randchoice(['normal','uniform'])
    scaler = QuantileTransformer(n_quantiles=a, output_distribution=c, subsample=b)
    tmp = str(a) + "_" + str(b) + "_" + c+ "_"+QuantileTransformer.__name__
    return scaler, tmp

def normalizer():
    a = _randchoice(['l1', 'l2','max'])
    scaler=Normalizer(norm=a)
    tmp=a+"_"+Normalizer.__name__
    return scaler,tmp

## IQR parameter
def binarize():
    a=_randuniform(0,100)
    scaler = Binarizer(threshold=a)
    tmp = str(round(a,4)) + "_" +Binarizer.__name__
    return scaler, tmp

def polynomial():
    a = _randint(2,10)
    b = _randchoice([True, False])
    c = _randchoice([True, False])
    scaler=PolynomialFeatures(degree=a, interaction_only=b, include_bias=c)
    tmp = str(a) + "_" + str(b) + "_" + str(c)+ "_"+PolynomialFeatures.__name__
    return scaler, tmp


class no_transform:

    def get_params(self):
        return {}

def no_transformation():
    return no_transform.__name__, no_transform.__name__

def transform(df,scaler):
    if scaler == no_transform.__name__:
        if "DataFrame" in str(type(df)):
            return df
        else:
            return pd.DataFrame(df)
    elif "DataFrame" in str(type(df)):

        copyDF = df.copy(deep=True)
        buggy_list = copyDF['Buggy'].values.tolist()
        tempDF = copyDF.drop(labels=['Buggy'], axis=1)
        df1 = pd.DataFrame(scaler.fit_transform(tempDF.values))
        df1['Buggy'] = buggy_list

        return df1
    elif "array" in str(type(df)):
        df1 = pd.DataFrame(scaler.fit_transform(df))
        return df1