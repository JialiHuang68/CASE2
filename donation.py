# %%
import pyspark as py
import pyspark.sql.types as typ
import pyspark.sql.functions as func
import pyspark.ml.feature as ft

conf = py.SparkConf().setAppName('test')
sc = py.SparkContext(conf=conf)

# %%
# labels = [
#     ('TARGET_B', typ.IntegerType()),
#     ('ID', typ.IntegerType()),
#     ('TARGET_D', typ.IntegerType()),
#     ('GiftCnt36', typ.IntegerType()),
#     ('GiftCntAll', typ.IntegerType()),
#     ('GiftCntCard36', typ.IntegerType()),
#     ('GiftCntCardAll', typ.IntegerType()),
#     ('GiftAvgLast', typ.FloatType()),
#     ('GiftAvg36', typ.FloatType()),
#     ('GiftAvgAll', typ.FloatType()),
#     ('GiftAvgCard36', typ.FloatType()),
#     ('GiftTimeLast', typ.IntegerType()),
#     ('GiftTimeFirst', typ.IntegerType()),
#     ('PromCnt12', typ.IntegerType()),
#     ('PromCnt36', typ.IntegerType()),
#     ('PromCntAll', typ.IntegerType()),
#     ('PromCntCard12', typ.IntegerType()),
#     ('PromCntCard36', typ.IntegerType()),
#     ('PromCntCardAll', typ.IntegerType()),
#     ('StatusCat96NK', typ.StringType()),
#     ('StatusCatStarAll', typ.IntegerType()),
#     ('DemCluster', typ.IntegerType()),
#     ('DemAge', typ.IntegerType()),
#     ('DemGender', typ.StringType()),
#     ('DemHomeOwner', typ.StringType()),
#     ('DemMedHomeValue', typ.IntegerType()),
#     ('DemPctVeterans', typ.IntegerType()),
#     ('DemMedIncome', typ.IntegerType())]

# schema = typ.StructType([typ.StructField(e[0], e[1], True) for e in labels])
data = py.SQLContext(sc).read.csv(
    'hdfs://master:9000/user/stm/test/donations2.csv',
    header=True,
    nullValue=False)
# 剔除 'ID', 'TARGET_D' 两列
data = data.drop('ID', 'TARGET_D')
data.describe().show()
# 总列数为9686 其中 GiftAvgCard36 为7906 DemAge 为7279

# %% GiftAvgCard36 以中位数补充
import numpy as np
list_GiftAvgCard36 = data.select('GiftAvgCard36').rdd.map(
    lambda row: row['GiftAvgCard36']).collect()
list_DemAge = data.select('DemAge').rdd.map(
    lambda row: row['DemAge']).collect()


def find_median(values_list):
    '''获取list的中位数'''
    try:
        # 剔除list中的None, 将list中的值转为float
        values_list = [float(x) for x in values_list if x is not None]
        median = np.nanmedian(values_list)
        return round(float(median), 2)
    except Exception:
        return None


median_GiftAvgCard36 = find_median(list_GiftAvgCard36)  # 12.5
data = data.fillna({'GiftAvgCard36': '12.5'})

median_DemAge = find_median(list_DemAge)  # 60
data = data.fillna({'DemAge': '60'})

# %% 转换columns的类型
# def str2int(col):
#     return func.col(col).cast(typ.IntegerType())

# def str2float(col):
#     return func.col(col).cast(typ.FloatType())

col_int = ['TARGET_B', 'GiftCnt36', 'GiftCntAll', 'GiftCntCard36', \
    'GiftCntCardAll', 'GiftTimeLast', 'GiftTimeFirst', 'PromCnt12', \
    'PromCnt36', 'PromCntAll', 'PromCntCard12', 'PromCntCard36', \
    'PromCntCardAll', 'StatusCatStarAll', 'DemCluster', \
    'DemPctVeterans']

col_float = ['GiftAvgLast', 'GiftAvg36', 'GiftAvgAll', \
    'GiftAvgCard36', 'DemAge', 'DemMedHomeValue', 'DemMedIncome']

for col in col_int:
    data = data.withColumn(col, data[col].cast(typ.IntegerType()))


for col in col_float:
    data = data.withColumn(col, data[col].cast(typ.FloatType()))

# %% woe变换
def calc_woe(data, x, y, qnt_num=20, min_block_size=20, v_type='c'):
    data.createOrReplaceTempView('woeTable')
    y_list = data.orderBy(y).groupby(y).count().collect()
    bad_num = y_list[0]['count']
    good_num = y_list[1]['count']
    woe_list = []
    iv_list = []
    if v_type == 'c':
        size = data.count()
        tmpTable = py.SQLContext(sc).sql("select row_number() over (order by " + x + ") as rnk, " + x + "," + y + " from woeTable")
        tmpTable.createOrReplaceTempView('tmpTable')
        bucket_size = np.ceil(size/qnt_num)
        for i in range(qnt_num):
            tmp = py.SQLContext(sc).sql("select * from tmpTable where rnk between " + str(i*bucket_size) + " and " + str((i+1)*bucket_size) + "")
            tmp_list = tmp.orderBy(y).groupBy(y).count().collect()
            bad = tmp_list[0]['count']
            good = tmp_list[1]['count']
            woe = np.log((good/good_num)/(bad/bad_num))
            woe_list.append(woe)
            iv = woe * ((good/good_num) - (bad/bad_num))
            iv_list.append(iv)
        return sum(iv_list)
    elif v_type == 'd':
        x_list = data.select(x).distinct().collect()
        for i in range(len(x_list)):
            tmp_list = data.filter("" + x + " == '" + x_list[i][x] + "'").orderBy(y).groupBy(y).count().collect()
            bad = tmp_list[0]['count']
            good = tmp_list[1]['count']
            woe = np.log((good/good_num)/(bad/bad_num))
            woe_list.append(woe)
            iv = woe * ((good/good_num) - (bad/bad_num))
            iv_list.append(iv)
    
    return sum(iv_list)

