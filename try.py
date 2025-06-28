import pickle

data_dict = pickle.load(open('data.pickle', 'rb'))

print("Type of data_dict['data']:", type(data_dict['data']))
print("Number of samples:", len(data_dict['data']))

# Check first sample
if len(data_dict['data']) > 0:
    print("Type of first sample:", type(data_dict['data'][0]))
    print("Length of first sample:", len(data_dict['data'][0]))
    print("First 5 values:", data_dict['data'][0][:5])