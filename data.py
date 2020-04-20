import csv


numerical    = num = "num"
numerical_id = nid = "nid"
categorical  = cat = "cat"
ignore       = ign = "ign"

target_numid = t_i = "t_i" 
target_num   = t_n = "t_n"
target_cat   = t_c = "t_c"

target_lab   = t_l = "t_l" # sklearn svm wants targets as 0, 1, 2, 3 etc. 


raw_headers = None
raw_data = None
headers = None
values = None
headers_t = None
targets = None

# TODO In future or as needed, create a test and train data split, although this is mostly a convenience, as you can just create two files 

def LoadDataTypes(file_name, encoding='utf-8'):
    with open(file_name, 'r', encoding=encoding) as d:
        data_reader = csv.reader(d)
        data_types = next(data_reader)

    return [d.strip() for d in data_types]

# print(LoadDataTypes('heart.format.csv'))

def LoadCSV(file_name, encoding='utf-8'):
    data = []
    header = []
    with open(file_name, 'r', encoding=encoding) as data_file:
        data_reader = csv.reader(data_file)
        header = next(data_reader)
        for row in data_reader:
            data.append(list(row))

    return header, data

# print(LoadCSV('heart.csv', encoding='utf-8-sig'))

'''
Note if passing categories in, must pass categories for every column
If none passed in, assume categorical
'''
def OneHotEncode(data, categories=[], headers=[], size_check=False):
    values = [] # range of values for all cols
    maps = []
    headers_extended = []
    target_headers = [] # for now just assume targets are in the range [0,1]
    targets = []

    if categories == []:
        # Assume all is categorical
        categories = [categorical] * len(data[0])
    if headers == []:
        headers = [''] * len(data[0])

    # print(categories)
    # pass 1, pick out range of values
    for i in range(len(categories)):
        col_values = [r[i] for r in data]
        headers_sub = []
        target_headers_sub = []
        if categories[i] == categorical:
            col_set = list(set(col_values))
            col_map = dict(list(zip(col_set, range(len(col_set)))))
            headers_sub = [headers[i]+'_'+c for c in col_set]

        elif categories[i] == numerical:
            col_nums = [float(x) for x in col_values]
            col_mean = sum(col_nums) / len(col_nums) 
            col_sd = (sum([(c-col_mean)**2 for c in col_nums])/(len(col_nums) - 1)) ** 0.5
            col_map = {"mean": col_mean, "sd": col_sd}
            headers_sub = [headers[i]]

        elif categories[i] == numerical_id:
            col_map = {}
            headers_sub = [headers[i]]

        elif categories[i] == target_numid:
            col_map = {}
            headers_sub = [headers[i]]

        elif categories[i] == target_num:
            col_nums = [float(x) for x in col_values]
            col_map = {"min": min(col_nums), "max": max(col_nums)}
            target_headers_sub = [headers[i]]

        elif categories[i] == target_cat:
            col_set = list(set(col_values))
            col_map = dict(list(zip(col_set, range(len(col_set)))))
            target_headers_sub = [headers[i]+'_'+c for c in col_set]

        elif categories[i] == target_lab:
            col_set = list(set(col_values))
            col_map = dict(list(zip(col_set, range(len(col_set)))))
            target_headers_sub = [headers[i]]

        elif categories[i] == ignore:
            pass

        else:
            raise ValueError(f"Category {categories[i]} given not recognised")

        maps.append(col_map)
        headers_extended += headers_sub
        target_headers += target_headers_sub

    if(size_check):
        return {'headers': headers_extended, 'n_features': len(headers_extended), 'n_cols': len(data)}
    
    # print(headers_extended)
    # pass 2, map each column to its one-hot representation
    for r in data:
        row_arr = []
        tar_arr = []
        for i in range(len(r)):
            v = r[i]
            c = categories[i]
            m = maps[i]
            val_arr = []
            tar = []
            if c == categorical:
                val_arr = [0] * len(m)
                val_arr[m[v]] = 1
            
            elif c == numerical_id:
                val_arr = [float(v)]
            
            elif c == numerical:
                val_arr = [(float(v) - m['mean']) / m['sd']]

            elif c == target_numid:
                tar = [float(v)]

            elif c == target_num: # needs updating to stddev or add option for both
                tar =  [(float(v)-m['min']) / (m['max'] - m['min'])]

            elif c == target_cat:
                tar = [0] * len(m)
                tar[m[v]] = 1

            elif c == target_lab:
                tar = [m[v]] # assume that target_lab is exclusively used, append values only

            elif c == ignore:
                pass

            row_arr += val_arr
            tar_arr += tar

        # print(row_arr)
        values.append(row_arr)
        targets.append(tar_arr)

    return headers_extended, values, target_headers, targets

# Must be called to create the classification data environment
def LoadEnvironment(data_file, form_file, encoding_data="utf-8"):
    global raw_headers, raw_data, headers, values, headers_t, targets
    form = LoadDataTypes(form_file)
    raw_headers_local, raw_data_local = LoadCSV(data_file, encoding=encoding_data)
    headers, values, headers_t, targets = OneHotEncode(raw_data_local, form, raw_headers_local)

    raw_headers, raw_data = raw_headers_local, raw_data_local

# print(OneHotEncode(raw_data, form, raw_headers,size_check=True))

# print(headers)
# for v in vals[:2]:
#     print(v)
# print(tar_headers)
# for t in tars[:2]:
#     print(t)


# print(OneHotEncode(data, categories=[cat,ign,cat,tno,tno], headers=['name1', 'name2', 'yes', 'nums','class']))

# raw        = []
# normalised = []
# headers    = []
# targets    = []
# lows       = []
# highs      = []
# target     = -1 # index as data[target] 

# # holdout
# train      = []
# train_t    = []
# test       = []
# test_t     = []
# n_train    = 0
# n_test     = 0

# with open('heart.csv', 'r') as file_in:
#     heart_data = csv.reader(file_in)

#     headers = next(heart_data)

#     for patient in heart_data:
#         raw.append([float(x) for x in patient[:target]])
#         targets.append(int(patient[target]))

# for i in range(len(headers) - 1):
#     col = [r[i] for r in raw]
#     lows.append(min(col))
#     highs.append(max(col))

# def regularise(r):
#     return [(r[i] - lows[i]) / (highs[i]-lows[i])  for i in range(len(r))]

# for r in raw:
#     normalised.append(regularise(r))


# def split(s=0.8):
#     # reset 
#     global train, train_t, test, test_t
#     train = []
#     train_t = []
#     test = []
#     test_t = []

#     # return indices of test and training set
#     full = list(range(len(normalised)))
#     n = int(s * len(normalised))
#     random.shuffle(full)
    
#     train_i = full[:n]
#     test_i  = full[n:]
    
#     for i in train_i:
#         train.append(normalised[i])
#         train_t.append(targets[i])

#     for i in test_i:
#         test.append(normalised[i])
#         test_t.append(targets[i])

# split()