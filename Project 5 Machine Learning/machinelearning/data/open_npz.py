from numpy import load

data = load('lang_id.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])

data = load('mnist.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])